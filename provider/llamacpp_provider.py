from .base_provider import BaseProvider
import torch
from typing import List
from llama_cpp import Llama, BaseLlamaCache

class LlamacppProvider(BaseProvider):
    def __init__(
        self,
        llm: Llama,
        cache: BaseLlamaCache
    ):
        self.llm = llm
        self.llm.set_cache(cache)

    def encode(self, text: str, add_special_tokens: bool=True) -> List[int]:
        return self.llm.tokenize(text.encode("utf-8", errors="ignore"), add_bos=add_special_tokens, special=add_special_tokens)

    def decode(self, tokens: List[int]) -> str:
        return self.llm.detokenize(tokens).decode("utf-8", errors="ignore")

    def generate(self, input_ids: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        pass

    def get_eos_token_id(self) -> int:
        return self.llm.token_eos()

    def crop_cache(self, idx: int) -> None:
        if idx >= 0:
            return; 
        while(idx < 0):
            self.llm.cache.cache_state.popitem(last=True)
            idx += 1

    def on_finish(self) -> None:
        del self.llm.cache
        size = self.llm.cache.cache_size()
        self.llm.set_cache(type(self.llm.cache)(size))
