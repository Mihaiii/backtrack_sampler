from .base_provider import BaseProvider
import torch
import numpy as np
import warnings
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
        # We access the private _logits_all attribute as there's no public API to set this post-initialization
        # This is necessary for the backtrack sampler to get logits for custom sampling
        if hasattr(self.llm, '_logits_all'):
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
        if not hasattr(self.llm, 'eval_logits'):
            raise RuntimeError("The llama-cpp-python version doesn't support eval_logits. Please update to a newer version.")
        
        logits_list = list(self.llm.eval_logits)
        if len(logits_list) == 0:
            raise RuntimeError("No logits available. Ensure the Llama model was initialized with logits_all=True and tokens have been evaluated.")
        logits = np.array(logits_list[-1], dtype=np.float32)
        return torch.from_numpy(logits).unsqueeze(0).to(self.device)

    def get_eos_token_id(self) -> int:
        return self.llm.token_eos()

    def remove_latest_cache(self, nr: int) -> None:
        # When we backtrack, we need to update our evaluated tokens count
        self._evaluated_tokens = max(0, self._evaluated_tokens - nr)
        
        # Also clear the KV cache for those tokens
        if nr > 0 and hasattr(self.llm, '_ctx') and hasattr(self.llm, 'n_tokens'):
            try:
                # Remove tokens from the llama context
                self.llm._ctx.kv_cache_seq_rm(-1, self.llm.n_tokens - nr, -1)
                self.llm.n_tokens -= nr
            except Exception as e:
                # If we can't clear the KV cache, log it but continue
                # This might happen if the llama-cpp-python internal API changes
                warnings.warn(f"Failed to clear KV cache during backtracking: {e}")
        
        # Clear from the cache as well
        while nr > 0:
            if len(self.llm.cache.cache_state) > 0:
                self.llm.cache.cache_state.popitem(last=True)
            nr -= 1

    def reset(self) -> None:
        self._evaluated_tokens = 0
        
        # Reset the token count and clear KV cache if available
        if hasattr(self.llm, 'n_tokens'):
            self.llm.n_tokens = 0
        
        if hasattr(self.llm, '_ctx'):
            try:
                self.llm._ctx.kv_cache_clear()
            except Exception as e:
                # If we can't clear the KV cache, log it but continue
                warnings.warn(f"Failed to clear KV cache during reset: {e}")
        
        # Also reset the cache
        new_cache = LlamaRAMCache(capacity_bytes=self.llm.cache.capacity_bytes)
        del self.llm.cache
        self.llm.set_cache(new_cache)
        
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
