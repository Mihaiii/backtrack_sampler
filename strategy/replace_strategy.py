import torch
import sys
from typing import List
from .antislop_strategy import AntiSlopStrategy
from ..provider.base_provider import BaseProvider


class ReplaceStrategy(AntiSlopStrategy):
    def __init__(
        self,
        provider: BaseProvider,
        find: List[str],
        replace: str,
        max_replacements: int = sys.maxsize,
        max_new_tokens_for_replace: int = sys.maxsize,
        skip_tokens: int = 0,
        max_prob: float = 1
    ):
        """
        find: The list of texts to be replaced.
        replace: The text to use as a replacement.
        max_replacements: Maximum number of replacements allowed.
        max_new_tokens_for_replace: Only allow find-and-replace within the first N generated tokens.
        skip_tokens: Only allow find-and-replace after N generated tokens.
        max_prob: Only allow find-and-replace if the probability of the first token in the found text is lower than N.
        """
        super().__init__(provider, find, with_variants=False, skip_tokens=skip_tokens)
        self.replace_ids = self.provider.encode(replace, add_special_tokens=False)
        self.max_replacements = max_replacements
        self.max_new_tokens_for_replace = max_new_tokens_for_replace
        self.max_prob = max_prob
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.replaced = 0
        self.replace_index = None

    def on_logits(
        self, logits: torch.Tensor, continuation_tokens: List[int]
    ) -> torch.Tensor:
        return logits

    def on_probs(
        self, probs: torch.FloatTensor, continuation_tokens: List[int]
    ) -> torch.FloatTensor:
        if self.slop_start_pos is not None:
            for slop_token in self.found_slop_tokens[self.slop_start_pos]:
                if probs[:, slop_token] <= self.max_prob:
                    self.replace_index = 0
                    self.replaced += 1

        if self.replace_index is not None:
            probs[:, self.replace_ids[self.replace_index]] = 1e9
            self.replace_index = self.replace_index + 1
            if self.replace_index >= len(self.replace_ids):
                self.replace_index = None
        return probs

    def backtrack(self, continuation_tokens: List[int]) -> List[int]:
        self.slop_start_pos = None
        if (
            self.max_new_tokens_for_replace > len(continuation_tokens)
            and self.max_replacements > self.replaced
            and self.skip_tokens <= len(continuation_tokens)
        ):
            return super().backtrack(continuation_tokens)
        return continuation_tokens
