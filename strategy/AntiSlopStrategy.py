import torch
from typing import List, Optional
from transformers import PreTrainedTokenizer
from .BacktrackStrategy import BacktrackStrategy

class AntiSlopStrategy(BacktrackStrategy):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        slop_phrase_prob_adjustments: List[str]
    ):
        self.tokenizer = tokenizer
        self.slop_phrase_prob_adjustments = slop_phrase_prob_adjustments
        self._checkpoint_index = 0

        self.tokenized_slops = self._tokenize_slop_variants()
        self.max_tokenized_slop = max(len(seq) for seq in self.tokenized_slops)

        self.logit_cache = []
        self.downregulated_positions = {}

    def get_checkpoint_index(self) -> int:
        return self._checkpoint_index
    
    def on_new_position_increment(self, current_position: int) -> None:
        self._checkpoint_index = max(current_position - self.max_tokenized_slop, 0)

    def backtrack(self, generated_sequence: list[int], current_position: int, past_key_values) -> tuple[list[int], int]:
        start_pos = self._detect_slops(generated_sequence[-self.get_checkpoint_index():])
        if start_pos is not None:
            initial_position = current_position

            while start_pos != current_position:
                generated_sequence.pop()
                current_position -= 1

            to_del = [key for key in self.logit_cache if key > start_pos]
            for key in to_del:
                del self.logit_cache[key]

            if past_key_values:
                past_key_values = tuple(tuple(layer[:, :, :current_position - initial_position, :] for layer in kv_pair) for kv_pair in past_key_values)

        return generated_sequence, current_position, past_key_values

    def on_logits(self, logits: torch.Tensor, position: int) -> torch.Tensor:
        if position < len(self.logit_cache):
            cached_entry = self.logit_cache[position]
            cached_logits = cached_entry['logits']
            
            for token_id in cached_entry['skip']:
                cached_logits[:, token_id] = float('-inf')

            logits = cached_logits.clone()
        else:
            self.logit_cache.append({
                'logits': logits.clone(),
                'skip': []
            })

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