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
        device: torch.device = torch.device('cuda'),
        use_cache: bool = True
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.device = device
        self.use_cache = use_cache

    @torch.no_grad()
    def generate_stream(
        self,
        prompt: str,
        max_length: int = None,
        max_new_tokens: int = None,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
        min_p: float = None,
    ) -> Generator[List[int], None, None]:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        generated_sequence = input_ids[0].tolist()
        current_position = len(generated_sequence)
        num_new_tokens = 0
        past_key_values = None

        while True:
            if max_length is not None and len(generated_sequence) >= max_length:
                break
            if max_new_tokens is not None and num_new_tokens >= max_new_tokens:
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
                use_cache=self.use_cache,
                past_key_values=past_key_values if self.use_cache else None,
            )

            if self.use_cache:
                past_key_values = outputs.past_key_values

            next_token_logits = outputs.scores[0]
            next_token_logits = next_token_logits / temperature

            # Apply strategy-specific penalty
            next_token_logits = self.strategy.apply_penalty(next_token_logits, current_position)

            # Apply min_p, top-k and top-p filtering
            filtered_logits = self._filter_logits(next_token_logits, top_k, top_p, min_p)

            # Sample the next token
            probs = torch.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            generated_sequence.append(next_token)
            current_position += 1
            num_new_tokens += 1

            # Clean up the KV cache
            if self.use_cache:
                past_key_values = self.strategy.clean_kv_cache(past_key_values, current_position)

            yield generated_sequence

            if next_token == self.tokenizer.eos_token_id:
                break

            # Apply backtracking if necessary
            generated_sequence, current_position = self.strategy.backtrack(generated_sequence, current_position)

    def _filter_logits(self, logits: torch.FloatTensor, top_k: int, top_p: float, min_p: float) -> torch.FloatTensor:
        # Implement the logit filtering logic here (min_p, top_k, top_p)
        # This method remains the same as in the original implementation
        pass