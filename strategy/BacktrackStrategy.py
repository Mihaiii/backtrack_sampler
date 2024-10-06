from abc import ABC, abstractmethod
import torch

class BacktrackStrategy(ABC):
    @property
    @abstractmethod
    def past_distributions_to_keep(self) -> int:
        pass

    @abstractmethod
    def backtrack(self, generated_sequence: list[int], current_position: int) -> tuple[list[int], int]:
        pass

    @abstractmethod
    def apply_penalty(self, logits: torch.Tensor, position: int) -> torch.Tensor:
        pass

    @abstractmethod
    def clean_kv_cache(self, past_key_values: tuple, current_position: int) -> tuple:
        pass