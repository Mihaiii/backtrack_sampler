# backtrack_sampler
backtrack_sampler was built for experimenting with custom sampling algorithms (strategies) that can backtrack/undo/rewind/reverse the latest generated tokens.
 
The code is short, simple and easy to understand.
 
If you want to make your own sampling algorithm, create a new file in the `/strategy` folder. Remember to submit a PR with it! The more strategies we have to experiment with, the better.
 
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

## Usage

## Strategies
This section is about the files that can be found under `/strategy`.
Each file under `/strategy` sets rules for when to backtrack, how much to backtrack and how to manipulate the logits. Since this package is made for experimenting, we highly encourage you to make your own file that and set your own rules for backtracking.

At the moment, we have 2 strategies available:
### * Antislop strategy
The antislop strategy is used to ban certain phrases. Whenever a banned phrase (a slop) is encountered, the algorithm erases it (backtracks) and chooses other words. The algorithm used [antislop-sampler](https://github.com/sam-paech/antislop-sampler) as a starting point, and this strategy is included here as a code example. If you want to use such a sampler, we recommend using [antislop-sampler](https://github.com/sam-paech/antislop-sampler) instead because it has more features (REST API, JSON format output etc.)

### * Creative writing strategy
## Thanks / credit
- [Sam Paech](https://x.com/sam_paech) for making [antislop-sampler](https://github.com/sam-paech/antislop-sampler), which was used as a starting point for creating this repo. Some parts of the code are still from the original repo.