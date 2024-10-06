import torch
from .BacktrackStrategy import BacktrackStrategy

class AntiFlattenDistributionStrategy(BacktrackStrategy):
    def __init__(self, adjustment_strength: float = 1.0):
        self._past_distributions_to_keep = 0
        self.adjustment_strength = adjustment_strength
        self.logit_cache = {}

    @property
    def past_distributions_to_keep(self) -> int:
        return self._past_distributions_to_keep

    def backtrack(self, generated_sequence: list[int], current_position: int) -> tuple[list[int], int]:
        # Implement backtracking logic here
        # For now, we'll just return the sequence as-is
        return generated_sequence, current_position

    def apply_penalty(self, logits: torch.Tensor, position: int) -> torch.Tensor:
        if position in self.logit_cache:
            cached_logits = self.logit_cache[position]
            flattened_logits = logits.clone()
            flattened_logits[cached_logits > flattened_logits] *= self.adjustment_strength
            return flattened_logits
        return logits

    def clean_kv_cache(self, past_key_values: tuple, current_position: int) -> tuple:
        # Clean up the logits cache
        to_del = [key for key in self.logit_cache if key < current_position - self.past_distributions_to_keep]
        for key in to_del:
            del self.logit_cache[key]
        
        # Truncate past_key_values if necessary
        if past_key_values:
            return tuple(tuple(layer[:, :, max(0, current_position - self.past_distributions_to_keep):, :] for layer in kv_pair) for kv_pair in past_key_values)
        return past_key_values