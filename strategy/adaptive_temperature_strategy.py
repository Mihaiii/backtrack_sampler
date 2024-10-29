import torch
from typing import List
from .base_strategy import BaseStrategy

# https://colab.research.google.com/drive/18-2Z4TMua-nwgCpIZo0lsKL6RDxH5Bvo
# https://github.com/Pleias/Quest-Best-Tokens
class AdaptiveTemperatureStrategy(BaseStrategy):
    def __init__(
        self,
        poly_coeffs: torch.Tensor = torch.tensor([-0.037, 0.481, -2.3, 4.917, -1.791]),
    ):
        self.poly_coeffs = poly_coeffs
        self.reset()

    def reset(self) -> None:
        self.idx = 0

    def get_keep_index(self) -> int:
        return self.idx

    def on_logits(
        self, logits: torch.FloatTensor, continuation_tokens: List[int]
    ) -> torch.FloatTensor:
        logits = self.adaptive_temperature_softmax(logits)
        return logits

    def on_probs(
        self,
        probs: torch.FloatTensor,
        continuation_tokens: List[int]
    ) -> torch.FloatTensor:
        return probs

    def on_next_token(
        self, continuation_tokens: List[int], probs: torch.FloatTensor
    ) -> None:
        self.idx = len(continuation_tokens)

    def backtrack(self, continuation_tokens: List[int]) -> List[int]:
        return continuation_tokens

    def adaptive_temperature_softmax(self, logits):
        """
        Implement adaptive temperature softmax based on entropy
        """
        x = self.compute_entropy(torch.nn.functional.softmax(logits, dim=-1))
        beta = (
            self.poly_coeffs[0] * x**4
            + self.poly_coeffs[1] * x**3
            + self.poly_coeffs[2] * x**2
            + self.poly_coeffs[3] * x
            + self.poly_coeffs[4]
        )
        if x <= 0.5:
            beta = 1.0
        else:
            beta = max(beta, 1.0)
        return logits * beta

    def compute_entropy(self, probs):
        """
        Compute Shannon entropy of probability distribution
        """
        return -torch.sum(
            probs * torch.log(probs + 1e-9)
        )  # We add a very small value to avoid 0s.
