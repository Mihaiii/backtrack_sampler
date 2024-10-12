import torch
from .base_strategy import BaseStrategy
from ..provider.base_provider import BaseProvider
from typing import List


class CreativeWritingStrategy(BaseStrategy):
    def __init__(
        self, 
        provider: BaseProvider, 
        top_p_flat: float = 0.65, 
        top_k_threshold_flat: int = 9, 
        eos_penalty: float = 0.8
    ):
        """
        top_p_flat: How many top tokens' probabilities sum up to this number?
        top_k_threshold_flat: If top_k_threshold_flat <= number of tokens that make up the top_p_flat value,
            then the distribution is considered flattened. The higher top_k_threshold_flat is, the less often
            the algo will rollback.
        eos_penalty: One commmon issue with this strategy is that it selects the eos too early when we ban
            the most probable token. Therefore, we can apply a penalty to the eos token.
            Values is between 0 and 1 where 1 means no penalty and 0 means eos is never selected 
            (in which case the generation will stop via the max_length or max_new_tokens settings).
        """
        self.eos_token = provider.get_eos_token_id()
        self.top_p_flat = top_p_flat
        self.top_k_threshold_flat = top_k_threshold_flat
        self.eos_penalty = eos_penalty
        self.reset()

    def get_keep_index(self) -> int:
        return self._keep_index

    def on_logits(
        self, logits: torch.FloatTensor, continuation_tokens: List[int]
    ) -> torch.FloatTensor:
        # if we just backtracked, then make sure the natural highest probable token will be selected
        # else, make sure it won't
        if self._is_flat and self._backtrack_data != None:
            logits[:, self._backtrack_data[1]] = torch.finfo(logits.dtype).max
            self._backtrack_data = None
        else:
            max_index = torch.argmax(logits).item()
            if(max_index != self.eos_token):
                logits[:, max_index] = float("-inf")
                
        return logits

    def on_probs(
        self, probs: torch.FloatTensor, continuation_tokens: List[int]
    ) -> torch.FloatTensor:
        self._is_flat = self._is_distribution_flat(probs)
        probs[:, self.eos_token] = probs[:, self.eos_token] * self.eos_penalty
        return probs

    def on_next_token(
        self, continuation_tokens: List[int], probs: torch.FloatTensor
    ) -> None:
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
        num_top_tokens = (
            torch.searchsorted(cumulative_probs, self.top_p_flat).item() + 1
        )

        return self.top_k_threshold_flat <= num_top_tokens
    
    def reset(self) -> None:
        self._is_flat = False
        self._backtrack_data = None
        self._keep_index = 0
