import torch
from typing import List
from .base_strategy import BaseStrategy
from ..provider.base_provider import BaseProvider


class DebugStrategy(BaseStrategy):
    def __init__(self, provider: BaseProvider):
        self.provider = provider
        self.reset()

    def reset(self) -> None:
        self.idx = 0

    def get_keep_index(self) -> int:
        return self.idx

    def on_logits(
        self, logits: torch.FloatTensor, continuation_tokens: List[int]
    ) -> torch.FloatTensor:
        return logits

    def on_probs(
        self,
        probs: torch.FloatTensor,
        continuation_tokens: List[int]
    ) -> torch.FloatTensor:
        top_2_probs, top_2_indices = torch.topk(probs, 2)
        chars = self.provider.decode(top_2_indices.flatten().tolist())
        print(
            f"\nTop 2 chars: {chars}, top 2 probs: {top_2_probs.flatten().tolist()}. Selected:"
        )
        return probs

    def on_next_token(
        self, continuation_tokens: List[int], probs: torch.FloatTensor
    ) -> None:
        self.idx = len(continuation_tokens)

    def backtrack(self, continuation_tokens: List[int]) -> List[int]:
        return continuation_tokens
