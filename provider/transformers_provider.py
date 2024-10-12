from .base_provider import BaseProvider
import torch
from typing import List
from transformers import PreTrainedTokenizer, PreTrainedModel, DynamicCache


class TransformersProvider(BaseProvider):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: torch.device = torch.device('cuda')
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.past_key_values = DynamicCache()
        self.device = device

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

    def generate(self, input_ids: List[int], *args, **kwargs) -> torch.Tensor:
        input = torch.tensor([input_ids], device=self.device)
        outputs = self.model.generate(
            input,
            max_new_tokens=1,
            do_sample=False,
            temperature=1,
            pad_token_id=self.tokenizer.pad_token_id,
            num_return_sequences=1,
            return_dict_in_generate=True,
            output_scores=True,
            past_key_values=self.past_key_values,
            *args,
            **kwargs
        )

        self.past_key_values = outputs.past_key_values
        return outputs.scores[0]

    def get_eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    def crop_cache(self, idx: int) -> None:
        self.past_key_values.crop(idx)

    def reset(self) -> None:
        del self.past_key_values
        self.past_key_values = DynamicCache()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
