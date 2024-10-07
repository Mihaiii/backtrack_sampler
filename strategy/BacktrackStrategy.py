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
    def backtrack(self, generated_sequence: list[int], current_position: int) -> Tuple[List[int], int, Optional[Tuple[Tuple[torch.Tensor, ...], ...]]]:
        pass

    @abstractmethod
    def on_logits(self, logits: torch.FloatTensor, position: int) -> torch.FloatTensor:
        pass

    @abstractmethod
    def on_probs(self, probs: torch.FloatTensor, position: int) -> torch.FloatTensor:
        pass

    @abstractmethod
    def on_next_token(self, token: int, position: int) -> None:
        pass