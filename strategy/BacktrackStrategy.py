from abc import ABC, abstractmethod
import torch

class BacktrackStrategy(ABC):
    @abstractmethod
    def get_checkpoint_index(self) -> int:
        pass

    @abstractmethod
    def on_new_position_increment(self) -> None:
        pass

    @abstractmethod
    def backtrack(self, generated_sequence: list[int], current_position: int) -> tuple[list[int], int]:
        pass

    @abstractmethod
    def on_logits(self, logits: torch.Tensor, position: int) -> torch.Tensor:
        pass