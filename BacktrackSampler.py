import torch
from transformers import PreTrainedTokenizer, PreTrainedModel
from typing import List, Generator
from strategy.BacktrackStrategy import BacktrackStrategy

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
        top_k: int = None,
        top_p: float = None,
        min_p: float = None,
        use_cache: bool = True
    ) -> Generator[List[int], None, None]:
        
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        generated_sequence = input_ids[0].tolist()
        yield generated_sequence
        current_position = 0
        released_position = 0
        past_key_values = None

        while True:
            if max_length is not None and len(generated_sequence) >= max_length:
                if released_position < current_position:
                    for token in generated_sequence[released_position-current_position:]:
                        yield token                
                break
            
            if max_new_tokens is not None and current_position >= max_new_tokens:
                if released_position < current_position:
                    for token in generated_sequence[released_position-current_position:]:
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
                use_cache=use_cache,
                past_key_values=past_key_values if use_cache else None,
            )

            if use_cache:
                past_key_values = outputs.past_key_values

            next_token_logits = outputs.scores[0]
            next_token_logits = next_token_logits / temperature

            # Opportunity to apply strategy-specific penalty
            next_token_logits = self.strategy.on_logits(next_token_logits, current_position)
            
            # Apply min_p, top-k and top-p filtering
            filtered_logits = self._filter_logits(next_token_logits, top_k, top_p, min_p)

            probs = torch.softmax(filtered_logits, dim=-1)
            
            probs = self.strategy.on_probs(probs, current_position)
            
            next_token = torch.multinomial(probs, num_samples=1).item()

            self.strategy.on_next_token(next_token, current_position)
            
            generated_sequence.append(next_token)
            current_position += 1
            self.strategy.on_new_position_increment(current_position)

            if released_position < self.strategy.get_checkpoint_index():
                yield next_token
                released_position = self.strategy.get_checkpoint_index()

            if next_token == self.tokenizer.eos_token_id:
                break

            # Apply backtracking if necessary
            generated_sequence, current_position, past_key_values = self.strategy.backtrack(generated_sequence, current_position, past_key_values)

        del past_key_values

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