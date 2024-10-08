import torch
from .backtrack_strategy import BacktrackStrategy
from typing import List, Optional, Tuple

class CreativeWritingStrategy(BacktrackStrategy):
    def __init__(self, cumulative_prob_threshold: float=0.8, min_to_be_flatten: int=3, min_second_highest_prob: float=0.25):
        self.cumulative_prob_threshold = cumulative_prob_threshold
        self.min_to_be_flatten = min_to_be_flatten
        self.min_second_highest_prob = min_second_highest_prob
        self._is_flat = False
        self._backtrack_position = None
        self._keep_index = 0

    def get_keep_index(self) -> int:
        return self._keep_index

    def on_logits(self, logits: torch.FloatTensor, continuation_tokens: List[int]) -> torch.FloatTensor:
        #if we just backtracked, then make the natural highest probable token the chosen one
        #else, make the chosen one the second natural highest probable token IF
        #the probability is >= min_second_highest_prob
        if self._is_flat and self._backtrack_position != None:
            logits[:, self._backtrack_position[1]] = torch.finfo(logits.dtype).max
            self._backtrack_position = None
        else:
            probabilities = torch.softmax(logits, dim=-1)
            probabilities = probabilities.view(-1)
            sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
            second_highest_prob = sorted_probs[1]
            if second_highest_prob >= self.min_second_highest_prob:
                logits[:, sorted_indices[1]] = torch.finfo(logits.dtype).max

        return logits

    def on_probs(self, probs: torch.FloatTensor, continuation_tokens: List[int]) -> torch.FloatTensor:
        self._is_flat = self._is_distribution_flat(probs)
        return probs

    def on_next_token(self, continuation_tokens: List[int], probs: torch.FloatTensor) -> None:
        latest_token = continuation_tokens[-1]
        highest_prob_token = torch.argmax(probs).item()

        if latest_token != highest_prob_token:
            self._keep_index = len(continuation_tokens) - 1
            self._backtrack_position = (self._keep_index, highest_prob_token)
        else:
            if self._backtrack_position is None:
                self._keep_index += 1

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
            
        return continuation_tokens, past_key_values

    def _is_distribution_flat(self, probs):
        """
        This answers the question:
        How many tokens are needed to get to a probability equal to the value of cumulative_prob_threshold.
        If that number of tokens is more than the value of min_to_be_flatten,
        then we consider we have a flatten distribution.
        """
        # Flatten probs to a 1D tensor
        probs = probs.view(-1)
    
        # Sort the probabilities in descending order
        sorted_probs, _ = torch.sort(probs, descending=True)
    
        # Compute the cumulative probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=0)
    
        # Find the number of top tokens needed to reach the cumulative probability threshold
        num_top_tokens = torch.searchsorted(cumulative_probs, self.cumulative_prob_threshold).item() + 1

        return self.min_to_be_flatten <= num_top_tokens