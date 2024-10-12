from abc import ABC, abstractmethod
from typing import List
import torch


class BaseProvider(ABC):
    @abstractmethod
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        pass

    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        pass

    @abstractmethod
    def generate(self, input_ids: List[int], *args, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def get_eos_token_id(self) -> int:
        pass

    @abstractmethod
    def remove_latest_cache(self, nr: int) -> None:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass
