import torch
from typing import List, Generator
from .strategy.base_strategy import BaseStrategy
from .provider.base_provider import BaseProvider


class BacktrackSampler:
    def __init__(
        self,
        strategy: BaseStrategy,
        provider: BaseProvider
    ):
        self.strategy = strategy
        self.provider = provider

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_length: int = None,
        max_new_tokens: int = None,
        temperature: float = 1.0,
        top_k: int = 50,  # same as HF's transformers default value
        top_p: float = None,
        min_p: float = None,
        *args,
        **kwargs
    ) -> Generator[List[int], None, None]:

        input_tokens = self.provider.encode(prompt, add_special_tokens=True)
        continuation_tokens = []
        release_index = 0

        while True:
            generated_sequence = input_tokens + continuation_tokens
            if max_length is not None and len(generated_sequence) >= max_length:
                for token in continuation_tokens[release_index:]:
                    yield token
                break

            if max_new_tokens is not None and len(continuation_tokens) >= max_new_tokens:
                for token in continuation_tokens[release_index:]:
                    yield token
                break

            outputs = self.provider.generate(
                generated_sequence, *args, **kwargs)

            next_token_logits = outputs / max(temperature, 1e-4)

            # Opportunity to apply strategy-specific penalty
            next_token_logits = self.strategy.on_logits(
                next_token_logits, continuation_tokens)

            # Apply min_p, top-k and top-p filtering
            filtered_logits = self._filter_logits(
                next_token_logits, top_k, top_p, min_p)

            probs = torch.softmax(filtered_logits, dim=-1)

            probs = self.strategy.on_probs(probs, continuation_tokens)

            next_token = torch.multinomial(probs, num_samples=1).item()

            continuation_tokens.append(next_token)
            self.strategy.on_next_token(continuation_tokens, probs)

            intial_len = len(continuation_tokens)
            # Apply backtracking if necessary
            continuation_tokens = self.strategy.backtrack(continuation_tokens)

            if (intial_len > len(continuation_tokens)):
                self.provider.crop_cache(len(continuation_tokens) - intial_len)

            while release_index < self.strategy.get_keep_index() - 1:
                yield continuation_tokens[release_index]
                release_index += 1

            if next_token == self.provider.get_eos_token_id():
                for token in continuation_tokens[release_index:]:
                    yield token
                break
        
        self.strategy.reset()
        self.provider.reset()

    def _filter_logits(self, logits: torch.FloatTensor, top_k: int, top_p: float, min_p: float) -> torch.FloatTensor:
        if min_p is not None:
            probs = torch.softmax(logits, dim=-1)
            top_prob, _ = torch.max(probs, dim=-1)
            scaled_min_p = min_p * top_prob
            logits = torch.where(probs < scaled_min_p, float('-inf'), logits)

        if top_k is not None and top_k > 0:
            top_k = min(top_k, logits.size(-1))
            top_k_logits, _ = torch.topk(logits, top_k)
            min_top_k = top_k_logits[:, -1].unsqueeze(-1)
            logits = torch.where(logits < min_top_k, float('-inf'), logits)

        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:,
                                     1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        return logits
