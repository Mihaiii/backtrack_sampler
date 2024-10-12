from .base_provider import BaseProvider
import torch
from typing import List
from llama_cpp import Llama, BaseLlamaCache


class LlamacppProvider(BaseProvider):
    def __init__(
        self,
        llm: Llama,
        cache: BaseLlamaCache,
        device: torch.device = torch.device('cpu')
    ):
        self.llm = llm
        self.llm.logits_all = True
        self.llm.context_params.logits_all = True
        self.llm.set_cache(cache)
        self.device = device

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        return self.llm.tokenize(text.encode("utf-8", errors="ignore"), add_bos=add_special_tokens, special=add_special_tokens)

    def decode(self, tokens: List[int]) -> str:
        return self.llm.detokenize(tokens).decode("utf-8", errors="ignore")

    def generate(self, input_ids: List[int], *args, **kwargs) -> torch.Tensor:
        prompt = self.decode(input_ids)
        output = self.llm(
            prompt,
            max_tokens=1,
            echo=False,
            temperature=1,
            top_p=1,
            top_k=9999999999999999,
            min_p=0,
            *args, **kwargs
        )
        logits = self.llm._scores[-1, :]
        return torch.from_numpy(logits).unsqueeze(0).to(self.device)

    def get_eos_token_id(self) -> int:
        return self.llm.token_eos()

    def crop_cache(self, idx: int) -> None:
        pass

    def on_finish(self) -> None:
        size = self.llm.cache.cache_size
        new_cache = type(self.llm.cache)(size)
        del self.llm.cache
        self.llm.set_cache(new_cache)
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
