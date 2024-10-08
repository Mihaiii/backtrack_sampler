import torch
from .backtrack_strategy import BacktrackStrategy
from typing import List, Optional, Tuple

class CreativeWritingStrategy(BacktrackStrategy):
    def __init__(self, top_p_flat: float=0.8, top_k_threshold_flat: int=3, min_prob_second_highest: float=0.25):
        """
        top_p_flat: How many top tokens' probabilities sum up to this number?
        top_k_threshold_flat: Minimum number of tokens required to sum up to top_p_flat in order
            for the distribution to be considered flatten. The higher top_k_threshold_flat is, the less often
            the algo will rollback.
        min_prob_second_highest: What is the minimum probability the second most probable token token must have
            in order to always be selected as next tokon, unless the rollback criterias will be met in the future.
        """
        self.top_p_flat = top_p_flat
        self.top_k_threshold_flat = top_k_threshold_flat
        self.min_prob_second_highest = min_prob_second_highest
        self._is_flat = False
        self._backtrack_position = None
        self._keep_index = 0

    def get_keep_index(self) -> int:
        return self._keep_index

    def on_logits(self, logits: torch.FloatTensor, continuation_tokens: List[int]) -> torch.FloatTensor:
        #if we just backtracked, then make the natural highest probable token the chosen one
        #else, make the chosen one the second natural highest probable token IF
        #the probability is >= min_prob_second_highest
        if self._is_flat and self._backtrack_position != None:
            logits[:, self._backtrack_position[1]] = torch.finfo(logits.dtype).max
            self._backtrack_position = None
        else:
            probabilities = torch.softmax(logits, dim=-1)
            probabilities = probabilities.view(-1)
            sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
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
            self._backtrack_position = (self._keep_index, highest_prob_token)
        else:
            if self._backtrack_position is None:
                self._keep_index += 1

    def backtrack(self, 
                  continuation_tokens: List[int],
                  past_key_values: Optional[Tuple[Tuple[torch.Tensor, ...], ...]]) -> Tuple[List[int], Optional[Tuple[Tuple[torch.Tensor, ...], ...]]]:
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
        How many tokens are needed to get to a probability equal to the value of top_p_flat.
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
        num_top_tokens = torch.searchsorted(cumulative_probs, self.top_p_flat).item() + 1

        return self.top_k_threshold_flat <= num_top_tokens