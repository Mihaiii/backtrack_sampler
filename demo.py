import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Union, Generator
from BacktrackSampler import BacktrackSampler
from strategy.AntiSlopStrategy import AntiSlopStrategy

model_name = "unsloth/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device('cuda')

slop_phrase_prob_adjustments = {
    'kaleidoscope': 0.5,
    'symphony': 0.5,
    'testament to': 0.5,
    'elara': 0.5,
    'moth to a flame': 0.5
}
prompt_text = "Write a story about Elara, the weaver of tapestries in future Technopolis. In the bustling city, a group of "
messages = [
    {"role": "user", "content": prompt_text}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
strategy = AntiSlopStrategy(tokenizer, slop_phrase_prob_adjustments)
sampler = BacktrackSampler(model, tokenizer, strategy, device)

token_stream = sampler.generate_stream(
    prompt=prompt,
    max_new_tokens=128,
    temperature=1
)

for tokens in token_stream:
    print(tokenizer.decode(tokens, skip_special_tokens=True))