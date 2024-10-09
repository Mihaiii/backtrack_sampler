from .backtrack_sampler_provider import BacktrackSamplerProvider
import torch
from typing import List

class LlamacppProvider(BacktrackSamplerProvider):
    def encode(self, text: str, add_special_tokens: bool=True) -> List[int]:
        pass

    def decode(self, tokens: List[int]) -> str:
        pass

    def generate(self, input_ids: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        pass

    def get_eos_token_id(self) -> int:
        pass

    def crop_cache(self, idx: int) -> None:
        pass

    def on_finish(self) -> None:
        pass
