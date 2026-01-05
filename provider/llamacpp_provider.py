from .base_provider import BaseProvider
import torch
import numpy as np
from typing import List
from llama_cpp import Llama, LlamaRAMCache


class LlamacppProvider(BaseProvider):
    def __init__(
        self,
        llm: Llama,
        cache: LlamaRAMCache,
        device: torch.device = torch.device("cpu"),
    ):
        self.llm = llm
        # Ensure logits_all is enabled - required for getting logits
        # If the Llama object wasn't created with logits_all=True, we set it here
        self.llm._logits_all = True
        self.llm.set_cache(cache)
        self.device = device
        self._evaluated_tokens = 0  # Track how many tokens have been evaluated

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        return self.llm.tokenize(
            text.encode("utf-8", errors="ignore"),
            add_bos=add_special_tokens,
            special=add_special_tokens,
        )

    def decode(self, tokens: List[int]) -> str:
        return self.llm.detokenize(tokens).decode("utf-8", errors="ignore")

    def generate(self, input_ids: List[int], *args, **kwargs) -> torch.Tensor:
        # Only evaluate new tokens that haven't been evaluated yet
        new_tokens = input_ids[self._evaluated_tokens:]
        
        if len(new_tokens) > 0:
            # Evaluate the new tokens
            self.llm.eval(new_tokens)
            self._evaluated_tokens = len(input_ids)
        
        # Get the logits for the last token
        # eval_logits returns a deque of logits, we want the last one
        logits_list = list(self.llm.eval_logits)
        if len(logits_list) == 0:
            raise RuntimeError("No logits available. Make sure the Llama model was initialized with logits_all=True")
        logits = np.array(logits_list[-1], dtype=np.float32)
        return torch.from_numpy(logits).unsqueeze(0).to(self.device)

    def get_eos_token_id(self) -> int:
        return self.llm.token_eos()

    def remove_latest_cache(self, nr: int) -> None:
        # When we backtrack, we need to update our evaluated tokens count
        self._evaluated_tokens = max(0, self._evaluated_tokens - nr)
        
        # Also clear the KV cache for those tokens
        if nr > 0:
            # Remove tokens from the llama context
            self.llm._ctx.kv_cache_seq_rm(-1, self.llm.n_tokens - nr, -1)
            self.llm.n_tokens -= nr
        
        # Clear from the cache as well
        while nr > 0:
            if len(self.llm.cache.cache_state) > 0:
                self.llm.cache.cache_state.popitem(last=True)
            nr -= 1

    def reset(self) -> None:
        self._evaluated_tokens = 0
        self.llm.n_tokens = 0  # Reset the token count
        self.llm._ctx.kv_cache_clear()  # Clear the KV cache
        
        # Also reset the cache
        new_cache = LlamaRAMCache(capacity_bytes=self.llm.cache.capacity_bytes)
        del self.llm.cache
        self.llm.set_cache(new_cache)
        
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
