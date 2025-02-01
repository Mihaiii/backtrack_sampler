import torch
from typing import List
from .base_strategy import BaseStrategy
from functools import reduce


class ChainStrategy(BaseStrategy):
    def __init__(self, strategies: List[BaseStrategy]):
        self.strategies = strategies

    def get_keep_index(self) -> int:
        return min([st.get_keep_index() for st in self.strategies])

    def on_logits(
        self, logits: torch.FloatTensor, continuation_tokens: List[int]
    ) -> torch.FloatTensor:
        return reduce(
            lambda res, strategy: strategy.on_logits(res, continuation_tokens),
            self.strategies,
            logits,
        )

    def on_probs(
        self, probs: torch.FloatTensor, continuation_tokens: List[int]
    ) -> torch.FloatTensor:
        return reduce(
            lambda res, strategy: strategy.on_probs(res, continuation_tokens),
            self.strategies,
            probs,
        )

    def on_next_token(
        self, continuation_tokens: List[int], probs: torch.FloatTensor
    ) -> None:
        for stg in self.strategies:
            stg.on_next_token(continuation_tokens, probs)

    def backtrack(self, continuation_tokens: List[int]) -> List[int]:
        return reduce(
            lambda res, strategy: strategy.backtrack(res),
            self.strategies,
            continuation_tokens,
        )

    def reset(self) -> None:
        for stg in self.strategies:
            stg.reset()
