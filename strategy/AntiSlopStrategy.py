import torch
from typing import List, Optional, Tuple
from transformers import PreTrainedTokenizer
from .BacktrackStrategy import BacktrackStrategy

class AntiSlopStrategy(BacktrackStrategy):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        slops: List[str]
    ):
        self.tokenizer = tokenizer
        self.slops = slops
        self._checkpoint_index = 0

        self.tokenized_slops = self._tokenize_slop_variants()
        self.max_tokenized_slop = max(len(seq) for seq in self.tokenized_slops)

        self.slop_start_pos = None

    def get_checkpoint_index(self) -> int:
        return self._checkpoint_index
    
    def on_new_position_increment(self, current_position: int) -> None:
        self._checkpoint_index = max(current_position - self.max_tokenized_slop, 0)

    def backtrack(self, 
                  generated_sequence: List[int], 
                  current_position: int, 
                  past_key_values: Optional[Tuple[Tuple[torch.Tensor, ...], ...]]) -> Tuple[List[int], int, Optional[Tuple[Tuple[torch.Tensor, ...], ...]]]:
        self.slop_start_pos = self._detect_slops(generated_sequence[-self._checkpoint_index:])
        if self.slop_start_pos is not None:
            initial_position = current_position

            while current_position > self.slop_start_pos:
                generated_sequence.pop()
                current_position -= 1

            if past_key_values:
                past_key_values = tuple(tuple(layer[:, :, :current_position - initial_position, :] for layer in kv_pair) for kv_pair in past_key_values)

        return generated_sequence, current_position, past_key_values

    def on_logits(self, logits: torch.Tensor, generated_sequence: List[int], position: int) -> torch.Tensor:
        if self.slop_start_pos is not None:
            logits[:, generated_sequence[self.slop_start_pos]] = float('-inf')
        return logits

    def _tokenize_slop_variants(self) -> list[list[int]]:
        token_sequences = []
        for word in self.slop_phrase_prob_adjustments.items():
            variants = [
                word.lower(),
                word.capitalize(),
                word.upper(),
                f" {word.lower()}",
                f" {word.capitalize()}",
                f" {word.upper()}",
            ]
            for variant in variants:
                token_ids = self.tokenizer.encode(variant, add_special_tokens=False)
                if token_ids:
                    token_sequences.append(token_ids)
        return token_sequences

    def _detect_slops(self, seq_since_checkpoint: List[int]) -> Optional[int]:
        min_index = None
        for slop in self.tokenized_slops:
            for i in range(len(seq_since_checkpoint) - len(slop) + 1):
                if seq_since_checkpoint[i:i+len(slop)] == slop:
                    if min_index is None or i < min_index:
                        min_index = i
                    break  # Found the first occurrence, move to next slop
        return min_index
    

    def on_probs(self, probs: torch.FloatTensor, generated_sequence: List[int], position: int) -> torch.FloatTensor:
        return probs

    def on_next_token(self, token: int, generated_sequence: List[int], position: int) -> None:
        pass