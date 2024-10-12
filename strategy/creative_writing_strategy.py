import torch
from .base_strategy import BaseStrategy
from typing import List


class CreativeWritingStrategy(BaseStrategy):
    def __init__(self, top_p_flat: float = 0.8, top_k_threshold_flat: int = 3, min_prob_second_highest: float = 0.25):
        """
        top_p_flat: How many top tokens' probabilities sum up to this number?
        top_k_threshold_flat: If top_k_threshold_flat <= number of tokens that make up the top_p_flat value, then the distribution is considered flattened. The higher top_k_threshold_flat is, the less often
            the algo will rollback.
        min_prob_second_highest: The minimum probability the second most probable token token must have
            in order to always be selected as next tokon.
        """
        self.top_p_flat = top_p_flat
        self.top_k_threshold_flat = top_k_threshold_flat
        self.min_prob_second_highest = min_prob_second_highest
        self._is_flat = False
        self._backtrack_data = None
        self._keep_index = 0

    def get_keep_index(self) -> int:
        return self._keep_index

    def on_logits(self, logits: torch.FloatTensor, continuation_tokens: List[int]) -> torch.FloatTensor:
        # if we just backtracked, then make the natural highest probable token the chosen one
        # else, make the chosen one the second natural highest probable token IF
        # its probability is >= min_prob_second_highest
        if self._is_flat and self._backtrack_data != None:
            logits[:, self._backtrack_data[1]] = torch.finfo(logits.dtype).max
            self._backtrack_data = None
        else:
            probabilities = torch.softmax(logits, dim=-1)
            probabilities = probabilities.view(-1)
            sorted_probs, sorted_indices = torch.sort(
                probabilities, descending=True)
            second_highest_prob = sorted_probs[1]
            if second_highest_prob >= self.min_prob_second_highest:
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
            self._backtrack_data = (self._keep_index, highest_prob_token)
        else:
            if self._backtrack_data is None:
                self._keep_index += 1

    def backtrack(self, continuation_tokens: List[int]) -> List[int]:
        if self._is_flat and self._backtrack_data != None:
            while len(continuation_tokens) > self._backtrack_data[0]:
                continuation_tokens.pop()

        return continuation_tokens

    def _is_distribution_flat(self, probs):
        """
        This answers the question:
        How many top tokens are needed to get to a probability equal to the value of top_p_flat.
        If that number of tokens is more than the value of top_k_threshold_flat,
        then we consider we have a flatten distribution.
        """
        # Flatten probs to a 1D tensor
        probs = probs.view(-1)

        # Sort the probabilities in descending order
        sorted_probs, _ = torch.sort(probs, descending=True)

        # Compute the cumulative probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=0)

        # Find the number of top tokens needed to reach the cumulative probability threshold
        num_top_tokens = torch.searchsorted(
            cumulative_probs, self.top_p_flat).item() + 1

        return self.top_k_threshold_flat <= num_top_tokens
