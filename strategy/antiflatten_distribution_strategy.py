import torch
from .backtrack_strategy import BacktrackStrategy
from typing import List, Optional, Tuple

class AntiFlattenDistributionStrategy(BacktrackStrategy):
    def __init__(self, entropy_threshold: float = 0.95):
        self.entropy_threshold = entropy_threshold
        self._is_flat = False
        self._backtrack_position = None
        self._release_index = 0

    def get_release_index(self) -> int:
        return self._release_index

    def backtrack(self, 
                  continuation_tokens: List[int],
                  past_key_values: Optional[Tuple[Tuple[torch.Tensor, ...], ...]]) -> Tuple[List[int], int, Optional[Tuple[Tuple[torch.Tensor, ...], ...]]]:
        if self._is_flat and self._backtrack_position != None:
            current_position = len(continuation_tokens)
            initial_position = current_position

            while current_position > self._backtrack_position[0]:
                continuation_tokens.pop()
                current_position -= 1

            if past_key_values:
                past_key_values = tuple(tuple(layer[:, :, :current_position - initial_position, :] for layer in kv_pair) for kv_pair in past_key_values)

            self._release_index = self._backtrack_position[0]
            
        return continuation_tokens, past_key_values

    def on_logits(self, logits: torch.FloatTensor, continuation_tokens: List[int]) -> torch.FloatTensor:
        if self._backtrack_position is not None:
            logits[:, self._backtrack_position[1]] = float('-inf')
            self._backtrack_position = None
        return logits

    def on_probs(self, probs: torch.FloatTensor, continuation_tokens: List[int]) -> torch.FloatTensor:
        self._is_flat = self._is_flat_distribution(probs)
        return probs

    def on_next_token(self, continuation_tokens: List[int], probs: torch.FloatTensor) -> None:
        latest_token = continuation_tokens[-1]
        highest_prob_token = torch.argmax(probs).item()

        if latest_token != highest_prob_token:
            self._release_index = len(continuation_tokens) - 1
            self._backtrack_position = (self._release_index, highest_prob_token)
        else:
            if self._backtrack_position is None:
                self._release_index += 1

    #Sonnet 3.5 generated code - ask it for details.
    def _is_flat_distribution(self, probs: torch.Tensor, slope_threshold: float = 0.1) -> bool:
        # Ensure we're working with CPU tensors
        probs = probs.cpu()

        # Sort probabilities in descending order
        sorted_probs, _ = torch.sort(probs, descending=True)

        # Calculate differences between adjacent probabilities
        diffs = sorted_probs[:-1] - sorted_probs[1:]

        # Find the first point where the slope is steeper than the threshold
        steep_point = torch.where(diffs > 0.1)[0]
        
        if len(steep_point) > 0:
            cutoff = steep_point[0].item() + 1  # +1 to include the steep point
        else:
            cutoff = len(sorted_probs)

        # Consider only the probabilities up to the cutoff
        flat_probs = sorted_probs[:cutoff]

        # Calculate the ratio of min to max in this range
        flatness_ratio = flat_probs[-1] / flat_probs[0]

        # Calculate the normalized entropy for this range
        entropy = -(flat_probs * torch.log2(flat_probs + 1e-10)).sum()
        max_entropy = torch.log2(torch.tensor(cutoff, dtype=torch.float32))
        normalized_entropy = entropy / max_entropy

        # Print debug information
        print(f"Cutoff point: {cutoff}")
        print(f"Flatness ratio: {flatness_ratio.item()}")
        print(f"Normalized Entropy: {normalized_entropy.item()}")

        # Determine if it's flat based on both ratio and entropy
        is_flat = flatness_ratio > 0.5 and normalized_entropy > 0.85

        return is_flat

