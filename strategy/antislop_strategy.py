import torch
from typing import List, Optional, Tuple
from transformers import PreTrainedTokenizer
from .backtrack_strategy import BacktrackStrategy

class AntiSlopStrategy(BacktrackStrategy):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        slops: List[str],
        release_index_buffer: int = 5
    ):
        self.tokenizer = tokenizer
        self.slops = slops
        self.release_index_buffer = release_index_buffer
        self._release_index = 0

        self.tokenized_slops = self._tokenize_slop_variants()
        self.max_tokenized_slop = max(len(seq) for seq in self.tokenized_slops)

        self.slop_start_pos = None
        # We need this in order to avoid an infinite loop where multiple different slops are generated from the same position
        # Basically we'll put all starting positions to -inf, not only the latest found.
        self.found_slop_tokens = {} 
    def get_release_index(self) -> int:
        return self._release_index
    
    def _update_release_index(self, continuation_tokens: List[int]) -> None:
        self._release_index = max(len(continuation_tokens) - self.max_tokenized_slop - self.release_index_buffer, 0)
        
    def on_next_token(self, continuation_tokens: List[int]) -> None:
        self._update_release_index(continuation_tokens)

    def backtrack(self, 
                  continuation_tokens: List[int],
                  past_key_values: Optional[Tuple[Tuple[torch.Tensor, ...], ...]]) -> Tuple[List[int], int, Optional[Tuple[Tuple[torch.Tensor, ...], ...]]]:
        self.slop_start_pos = self._detect_slops(continuation_tokens)
        current_position = len(continuation_tokens)
        if self.slop_start_pos is not None:
            self.found_slop_tokens.setdefault(self.slop_start_pos, set())
            self.found_slop_tokens[self.slop_start_pos].add(continuation_tokens[self.slop_start_pos])

            initial_position = current_position

            while current_position > self.slop_start_pos:
                continuation_tokens.pop()
                current_position -= 1

            if past_key_values:
                past_key_values = tuple(tuple(layer[:, :, :current_position - initial_position, :] for layer in kv_pair) for kv_pair in past_key_values)

            self._update_release_index(continuation_tokens)
            
        return continuation_tokens, past_key_values

    def on_logits(self, logits: torch.Tensor, continuation_tokens: List[int]) -> torch.Tensor:
        if self.slop_start_pos is not None:
            for token in self.found_slop_tokens[self.slop_start_pos]:
                logits[:, token] = float('-inf')
        return logits

    def _tokenize_slop_variants(self) -> list[list[int]]:
        token_sequences = []
        for slop in self.slops:
            variants = [
                slop.lower(),
                slop.capitalize(),
                slop.upper(),
                f" {slop.lower()}",
                f" {slop.capitalize()}",
                f" {slop.upper()}",
            ]
            for variant in variants:
                token_ids = self.tokenizer.encode(variant, add_special_tokens=False)
                if token_ids:
                    token_sequences.append(token_ids)
        return token_sequences

    def _detect_slops(self, tokens: List[int]) -> Optional[int]:
        min_index = None
        for slop in self.tokenized_slops:
            for i in range(len(tokens) - len(slop) + 1):
                if tokens[i:i+len(slop)] == slop:
                    if min_index is None or i < min_index:
                        min_index = i
                    break  # Found the first occurrence, move to next slop
        return min_index
    

    def on_probs(self, probs: torch.FloatTensor, continuation_tokens: List[int]) -> torch.FloatTensor:
        return probs