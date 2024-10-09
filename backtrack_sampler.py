import torch
from transformers import PreTrainedTokenizer, PreTrainedModel, DynamicCache
from typing import List, Generator
from strategy.backtrack_strategy import BacktrackStrategy

class BacktrackSampler:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        strategy: BacktrackStrategy,
        device: torch.device = torch.device('cuda')
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.device = device

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_length: int = None,
        max_new_tokens: int = None,
        temperature: float = 1.0,
        top_k: int = 50, #same as HF's transformers default value
        top_p: float = None,
        min_p: float = None,
        *args, 
        **kwargs
    ) -> Generator[List[int], None, None]:
        
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        input_tokens = input_ids[0].tolist()
        continuation_tokens = []
        release_index = 0
        past_key_values = DynamicCache()

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

            current_input_ids = torch.tensor([generated_sequence], device=self.device)

            outputs = self.model.generate(
                current_input_ids,
                max_new_tokens=1,
                do_sample=False,
                temperature=1,  # We apply temp ourselves after this
                pad_token_id=self.tokenizer.pad_token_id,
                num_return_sequences=1,
                return_dict_in_generate=True,
                output_scores=True,
                past_key_values=past_key_values,
                *args,
                **kwargs
            )

            past_key_values = outputs.past_key_values

            next_token_logits = outputs.scores[0] / max(temperature, 1e-8)

            # Opportunity to apply strategy-specific penalty
            next_token_logits = self.strategy.on_logits(next_token_logits, continuation_tokens)
            
            # Apply min_p, top-k and top-p filtering
            filtered_logits = self._filter_logits(next_token_logits, top_k, top_p, min_p)

            probs = torch.softmax(filtered_logits, dim=-1)
            
            probs = self.strategy.on_probs(probs, continuation_tokens)
            
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            continuation_tokens.append(next_token)
            self.strategy.on_next_token(continuation_tokens, probs)

            intial_len = len(continuation_tokens)
            # Apply backtracking if necessary
            continuation_tokens = self.strategy.backtrack(continuation_tokens)

            if(intial_len > len(continuation_tokens)):
                past_key_values.crop(len(continuation_tokens) - intial_len)
            
            while release_index < self.strategy.get_keep_index() - 1:
                yield continuation_tokens[release_index]
                release_index += 1

            if next_token == self.tokenizer.eos_token_id:
                for token in continuation_tokens[release_index:]:
                    yield token
                break

        del past_key_values
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

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
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        return logits
