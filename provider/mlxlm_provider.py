from .base_provider import BaseProvider
import torch
from typing import List
import mlx.nn as nn
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
from functools import partial
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
from mlx_lm.sample_utils import make_sampler
from mlx_lm.utils import stream_generate
import numpy as np

class MlxlmProvider(BaseProvider):
    def __init__(
        self,
        model: nn.Module,
        tokenizer: TokenizerWrapper
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_cache = make_prompt_cache(model)
        self.generator = partial(
            stream_generate,
            model=model,
            tokenizer=tokenizer,
            sampler=make_sampler(1.0, 1.0),
        )

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        # Why does tokenizer.convert_tokens_to_ids(text) exist?
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

    def generate(self, input_ids: List[int], *args, **kwargs) -> torch.Tensor:
        prompt = self.decode(input_ids)
        outputs = next(self.generator(
                    prompt=prompt,
                    prompt_cache=self.prompt_cache,
                    *args,
                    **kwargs
                ))
        logprobs_np = np.asarray(outputs.logprobs)
        logprobs_torch = torch.from_numpy(logprobs_np).to(torch.float32).unsqueeze(0)
        return logprobs_torch
    
    def get_eos_token_id(self) -> List[int]:
        return list(self.tokenizer._eos_token_ids)

    def remove_latest_cache(self, nr: int) -> None:
        trim_prompt_cache(self.prompt_cache, nr)

    def reset(self) -> None:
        del self.prompt_cache
        self.prompt_cache = make_prompt_cache(self.model)