import torch
from typing import List, Optional
from .base_strategy import BaseStrategy
from ..provider.base_provider import BaseProvider


class AntiSlopStrategy(BaseStrategy):
    def __init__(
        self,
        provider: BaseProvider,
        slops: List[str],
        keep_index_buffer: int = 5
    ):
        self.provider = provider
        self.slops = slops
        self.keep_index_buffer = keep_index_buffer

        self.tokenized_slops = self._tokenize_slop_variants()
        self.max_tokenized_slop = max((len(seq) for seq in self.tokenized_slops), default=0)

        self.reset()

    def get_keep_index(self) -> int:
        return self._keep_index

    def _update_keep_index(self, continuation_tokens: List[int]) -> None:
        self._keep_index = max(
            len(continuation_tokens) - self.max_tokenized_slop - self.keep_index_buffer, 0)

    def on_logits(self, logits: torch.Tensor, continuation_tokens: List[int]) -> torch.Tensor:
        if self.slop_start_pos is not None:
            for token in self.found_slop_tokens[self.slop_start_pos]:
                logits[:, token] = float('-inf')
        return logits

    def on_probs(self, probs: torch.FloatTensor, continuation_tokens: List[int]) -> torch.FloatTensor:
        return probs

    def on_next_token(self, continuation_tokens: List[int], probs: torch.FloatTensor) -> None:
        self._update_keep_index(continuation_tokens)

    def backtrack(self, continuation_tokens: List[int]) -> List[int]:
        self.slop_start_pos = self._detect_slops(continuation_tokens)
        if self.slop_start_pos is not None:
            self.found_slop_tokens.setdefault(self.slop_start_pos, set())
            self.found_slop_tokens[self.slop_start_pos].add(
                continuation_tokens[self.slop_start_pos])

            while len(continuation_tokens) > self.slop_start_pos:
                continuation_tokens.pop()

            self._update_keep_index(continuation_tokens)

        return continuation_tokens

    def _tokenize_slop_variants(self) -> list[list[int]]:
        token_sequences = []
        for slop in self.slops:
            variants = set([
                slop,
                slop.lower(),
                slop.capitalize(),
                slop.upper(),
                f" {slop.lower()}",
                f" {slop.capitalize()}",
                f" {slop.upper()}",
            ])
            for variant in variants:
                token_ids = self.provider.encode(
                    variant, add_special_tokens=False)
                if token_ids:
                    token_sequences.append(token_ids)
        return token_sequences

    def _detect_slops(self, tokens: List[int]) -> Optional[int]:
        min_index = None
        for slop in self.tokenized_slops:
            for i in range(len(tokens) - len(slop) + 1):
                if tokens[i:i+len(slop)] == slop:
                    if min_index is None or i < min_index:
                        min_index = i
                    break  # Found the first occurrence, move to next slop
        return min_index
    
    def reset(self) -> None:
        self._keep_index = 0
        self.slop_start_pos = None
        # We need this in order to avoid an infinite loop where multiple different slops 
        # are generated from the same position.
        # Basically we'll put all starting positions to -inf, not only the latest found.
        self.found_slop_tokens = {}
