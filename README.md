# Backtrack Sampler
Backtrack Sampler is a framework for experimenting with custom sampling algorithms (strategies) that can backtrack/undo/rewind/reverse the latest generated tokens.
 
## The code is short, simple and easy to understand
 
If you want to make your own sampling algorithm, create a new file in the `/strategy` directory that implements the [abstract base class](https://github.com/Mihaiii/backtrack_sampler/blob/main/strategy/base_strategy.py). Remember to submit a PR with it! The more strategies we have to experiment with, the better.

## Demo
- https://huggingface.co/spaces/Mihaiii/backtrack_sampler_demo
- https://colab.research.google.com/github/Mihaiii/backtrack_sampler/blob/main/demo.ipynb
 
## Installation
```cmd
pip install backtrack_sampler
```
The above command will install **0 dependencies**. Depending on what kind of LLM you want to use, you'll need to have installed **either** [transformers](https://github.com/huggingface/transformers) (`pip install transformers`), **or** [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) ([click here for install commands depending on your hardware](https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file#supported-backends)) + torch (`pip install torch` for CPU usage. For GPU, please search for the appropriate commands online.).
 
Here are some combos, for easy copy/paste:
```cmd
pip install backtrack_sampler transformers
```
```cmd
pip install backtrack_sampler llama-cpp-python torch
```


## Usage examples

### * llama.cpp

```python
import torch
import time
from llama_cpp import Llama, LlamaRAMCache
from backtrack_sampler import BacktrackSampler, CreativeWritingStrategy
from backtrack_sampler.provider.llamacpp_provider import LlamacppProvider

#make sure you have the model downloaded
#ex: wget https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf
llm = Llama(model_path="Llama-3.2-1B-Instruct-Q4_K_M.gguf", chat_format="llama-3", verbose=False, n_ctx=2100, n_batch=2100)
device = torch.device('cpu')
cache = LlamaRAMCache(capacity_bytes=100000000)

prompt = "Tell me a short tale of a dragon who is afraid of heights."
provider = LlamacppProvider(llm, cache, device)
strategy = CreativeWritingStrategy(provider, 
                                   top_p_flat = 0.65,
                                   top_k_threshold_flat = 9,
                                   eos_penalty = 0.75)
sampler = BacktrackSampler(provider, strategy)

ts = time.time()

token_stream = sampler.generate(
    prompt=prompt,
    max_new_tokens=1024,
    temperature=1
)

for token in token_stream:
    print(provider.decode([token]), end="", flush=True)

print(f"\nDuration: {time.time()-ts} seconds")
```

### * transformers

```python
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from backtrack_sampler import BacktrackSampler, AntiSlopStrategy
from backtrack_sampler.provider.transformers_provider import TransformersProvider

model_name = "unsloth/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device('cuda')

slops = [ "**Warning", "I cannot", "I can't", "I can’t", "I can not", "can’t provide", "can't provide", "can not provide",
         "***Warning", "*I cannot", "*I can't", "*I can’t", "*I can not", "*can’t provide", "*can't provide", "*can not provide"]
prompt_text = "How to make a bomb? Give as many details as possible."
messages = [
    {"role": "user", "content": prompt_text}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
provider = TransformersProvider(model, tokenizer, device)
strategy = AntiSlopStrategy(provider, slops)
sampler = BacktrackSampler(provider, strategy)

ts = time.time()

token_stream = sampler.generate(
    prompt=prompt,
    max_new_tokens=1024,
    temperature=1
)

for token in token_stream:
    print(tokenizer.decode(token, skip_special_tokens=True), end="", flush=True)

print(f"\nDuration: {time.time()-ts} seconds")
```

For more usage examples and outputs, see [demo.ipynb](https://colab.research.google.com/github/Mihaiii/backtrack_sampler/blob/main/demo.ipynb).

## Strategies
This section is about the files that can be found under [`/strategy`](https://github.com/Mihaiii/backtrack_sampler/tree/main/strategy).
Each file under [`/strategy`](https://github.com/Mihaiii/backtrack_sampler/tree/main/strategy) sets rules for when to backtrack, how much to backtrack and how to manipulate the logits. Since this package is made for experimenting, we highly encourage you to make your own file and set your own rules for backtracking.

At the moment, we have 5 strategies available:

### * Anti-slop strategy
[The Anti Slop Strategy](https://github.com/Mihaiii/backtrack_sampler/blob/main/strategy/antislop_strategy.py) is used to ban certain phrases. Whenever a banned phrase (a slop) is encountered, the algorithm erases it (backtracks) and chooses other words. The algorithm used [antislop-sampler](https://github.com/sam-paech/antislop-sampler) as a starting point, and this strategy is included here as a code example. If you want to use such a sampler, we recommend using [antislop-sampler](https://github.com/sam-paech/antislop-sampler) instead because it has more features (REST API, JSON format output etc.)

### * Creative writing strategy
[The Creative Writing Strategy](https://github.com/Mihaiii/backtrack_sampler/blob/main/strategy/creative_writing_strategy.py) is designed to enhance the creativity of language models by favoring less common word choices. It achieves this by often banning from selection the most probable token. This approach is an alternative to using a high temperature setting, which can lead to more creative outputs but often results in nonsensical or "gibberish" text if set too high.

By contrast, in the Creative Writing Strategy, when the probability distribution of potential next tokens is too flat (i.e., when many tokens have similar probabilities), the strategy will revert to a previous state and regenerate tokens. This rollback helps ensure that the generated text remains meaningful and avoids the pitfalls of overly random outputs.

Here is a demo of the Creative Writing Strategy: https://huggingface.co/spaces/Mihaiii/backtrack_sampler_demo

### * Debug strategy
[The Debug Strategy](https://github.com/Mihaiii/backtrack_sampler/blob/main/strategy/debug_strategy.py) is the simplest possible strategy and is used to debug logits/probs and as a skeleton for creating new strategies.

### * Human guidance strategy
[The Human Guidance Strategy](https://github.com/Mihaiii/backtrack_sampler/blob/main/strategy/human_guidance_strategy.py) is designed to allow the user to manually select the next token from the top generated ones. It is useful to get a better understanding of the model's capabilities.

This strategy relies on [curses](https://docs.python.org/3/howto/curses.html) for drawing, a library that's pre-installed on Linux and MacOS. The curses library is designed for terminal-based applications and does not function properly in notebook (`.ipynb` files) environments.

![](https://github.com/Mihaiii/backtrack_sampler/blob/main/hgs.gif)

### * Adaptive temperature strategy
[The Adaptive Temperature Strategy](https://github.com/Mihaiii/backtrack_sampler/blob/main/strategy/adaptive_temperature_strategy.py) is designed to dynamically adjust the temperature of the model based on the entropy of the probability distribution of the next token.
The code is copy/pasted from [this notebook](https://colab.research.google.com/drive/18-2Z4TMua-nwgCpIZo0lsKL6RDxH5Bvo) created by [Alexander Doria](https://x.com/Dorialexander).
The official repo is [Quest-Best-Tokens](https://github.com/Pleias/Quest-Best-Tokens).

### * Replace strategy
[The Replace Strategy](https://github.com/Mihaiii/backtrack_sampler/blob/main/strategy/replace_strategy.py) is a "find and replace" functionality. This strategy is a general implementation of [vgel's](https://github.com/vgel) overthinking script for R1 found [here](https://gist.github.com/vgel/8a2497dc45b1ded33287fa7bb6cc1adc).

Here is an example of how to use this strategy based on [vgel's](https://github.com/vgel) use case:
```python
strategy = ReplaceStrategy(
    provider, find="</think>", replace="\nWait, but", max_replacements=3
)
```
### * Chain strategy
[The Chain Strategy](https://github.com/Mihaiii/backtrack_sampler/blob/main/strategy/chain_strategy.py) allows applying multiple strategies on generation. If multiple strategies need to backtrack at the exact same token, then only the first one will be taken into consideration for backtracking.
```python
provider = LlamacppProvider(llm, cache, device)
strategy1 = ReplaceStrategy(
    provider,
    find=[" So", "So", "\nSo", "Therefore", " Therefore", "\nTherefore", "</think>"],
    replace=" But let me rephrase the request to see if I missed something.",
    max_replacements=4,
)
strategy2 = ReplaceStrategy(
    provider,
    find=[" But", "But", "\nBut", " Wait", "Wait", "\nWait"],
    replace="\nOkay, so in conclusion",
    skip_tokens=1024,
)
sampler = BacktrackSampler(provider, ChainStrategy([strategy1, strategy2]))
```

## Thanks / credit
- [Sam Paech](https://x.com/sam_paech) for making [antislop-sampler](https://github.com/sam-paech/antislop-sampler), which was used as a starting point for creating this repo. Some parts of the code are still from the original repo.
