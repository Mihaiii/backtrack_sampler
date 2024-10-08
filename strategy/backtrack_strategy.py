import torch
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

class BacktrackStrategy(ABC):
    @abstractmethod
    def get_release_index(self) -> int:
        """
        Everything up until (but not included) this index (meaning it starts with 0)
        is considered valid token generation that can't be rollbacked and therefore
        can be sent to the client for its use.
        """
        pass

    @abstractmethod
    def backtrack(self, continuation_tokens: list[int]) -> Tuple[int, Optional[Tuple[Tuple[torch.Tensor, ...], ...]]]:
        pass

    @abstractmethod
    def on_logits(self, logits: torch.FloatTensor, continuation_tokens: List[int]) -> torch.FloatTensor:
        pass

    @abstractmethod
    def on_probs(self, probs: torch.FloatTensor, continuation_tokens: List[int]) -> torch.FloatTensor:
        pass

    @abstractmethod
    def on_next_token(self, continuation_tokens: List[int], probs: torch.FloatTensor) -> None:
        pass