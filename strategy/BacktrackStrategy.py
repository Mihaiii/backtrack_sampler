import torch
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

class BacktrackStrategy(ABC):
    @abstractmethod
    def get_checkpoint_index(self) -> int:
        pass

    @abstractmethod
    def on_new_position_increment(self) -> None:
        pass

    @abstractmethod
    def backtrack(self, continuation_tokens: list[int], current_position: int) -> Tuple[int, Optional[Tuple[Tuple[torch.Tensor, ...], ...]]]:
        pass

    @abstractmethod
    def on_logits(self, logits: torch.FloatTensor, continuation_tokens: List[int], position: int) -> torch.FloatTensor:
        pass

    @abstractmethod
    def on_probs(self, probs: torch.FloatTensor, continuation_tokens: List[int], position: int) -> torch.FloatTensor:
        pass

    @abstractmethod
    def on_next_token(self, token: int, continuation_tokens: List[int], position: int) -> None:
        pass