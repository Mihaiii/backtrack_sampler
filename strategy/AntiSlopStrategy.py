import torch
from typing import Dict, Tuple, Set
from transformers import PreTrainedTokenizer
from .BacktrackStrategy import BacktrackStrategy

class AntiSlopStrategy(BacktrackStrategy):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        slop_phrase_prob_adjustments: Dict[str, float],
        adjustment_strength: float = 1.0
    ):
        self.tokenizer = tokenizer
        self.slop_phrase_prob_adjustments = slop_phrase_prob_adjustments
        self.adjustment_strength = adjustment_strength
        self._past_distributions_to_keep = 0

        self.token_sequences = self._prepare_token_sequences()
        self.max_sequence_length = max(len(seq) for seq in self.token_sequences.keys())
        self.starting_tokens_lookup = self._precompute_starting_tokens()

        self.logit_cache = {}
        self.downregulated_positions = {}

    @property
    def past_distributions_to_keep(self) -> int:
        return self._past_distributions_to_keep

    def backtrack(self, generated_sequence: list[int], current_position: int) -> tuple[list[int], int]:
        matched_sequence, start_pos = self._detect_disallowed_sequence(generated_sequence)
        if matched_sequence:
            # Backtrack: remove tokens from the generated_sequence that are part of the disallowed sequence
            for _ in range(len(matched_sequence)):
                generated_sequence.pop()
                current_position -= 1

            # Clear the logit_cache ahead of start_pos since we've backtracked
            to_del = [key for key in self.logit_cache if key > start_pos]
            for key in to_del:
                del self.logit_cache[key]

        return generated_sequence, current_position

    def apply_penalty(self, logits: torch.Tensor, position: int) -> torch.Tensor:
        if position in self.logit_cache:
            cached_logits = self.logit_cache[position]
            
            # Apply downregulation for slop phrases
            if position in self.downregulated_positions:
                for sequence in self.downregulated_positions[position]:
                    adjustment = self.token_sequences[sequence]
                    starting_tokens = self.starting_tokens_lookup.get(sequence, set())
                    for token_id in starting_tokens:
                        cached_logits[:, token_id] *= adjustment ** self.adjustment_strength

            logits = cached_logits.clone()
        else:
            self.logit_cache[position] = logits.clone()

        return logits

    def clean_kv_cache(self, past_key_values: tuple, current_position: int) -> tuple:
        # Clean up the logits cache
        to_del = [key for key in self.logit_cache if key < current_position - self.past_distributions_to_keep]
        for key in to_del:
            del self.logit_cache[key]
        
        # Truncate past_key_values if necessary
        if past_key_values:
            return tuple(tuple(layer[:, :, max(0, current_position - self.past_distributions_to_keep):, :] for layer in kv_pair) for kv_pair in past_key_values)
        return past_key_values

    def _prepare_token_sequences(self) -> Dict[Tuple[int, ...], float]:
        token_sequences = {}
        for word, prob_adjustment_factor in self.slop_phrase_prob_adjustments.items():
            variants = [
                word.lower(),
                word.capitalize(),
                word.upper(),
                f" {word.lower()}",
                f" {word.capitalize()}",
                f" {word.upper()}",
            ]
            for variant in variants:
                token_ids = tuple(self.tokenizer.encode(variant, add_special_tokens=False))
                if token_ids:
                    token_sequences[token_ids] = prob_adjustment_factor
        return token_sequences

    def _precompute_starting_tokens(self) -> Dict[Tuple[int, ...], Set[int]]:
        starting_tokens_lookup = {}
        for word in self.slop_phrase_prob_adjustments.keys():
            variants = [
                word.lower(),
                word.capitalize(),
                word.upper(),
                f" {word.lower()}",
                f" {word.capitalize()}",
                f" {word.upper()}",
            ]
            for variant in variants:
                token_ids = self.tokenizer.encode(variant, add_special_tokens=False)
                starting_tokens = set()
                if token_ids:
                    starting_tokens.add(token_ids[0])
                    first_token_decoded = self.tokenizer.decode(token_ids[0], skip_special_tokens=True)
                    for i in range(len(first_token_decoded) - 1):
                        prefix = first_token_decoded[:-(i + 1)]
                        encoded_prefix = self.tokenizer.encode(prefix, add_special_tokens=False)
                        if encoded_prefix:
                            starting_tokens.add(encoded_prefix[0])
                    starting_tokens_lookup[tuple(token_ids)] = starting_tokens
        return starting_tokens_lookup

    def _detect_disallowed_sequence(self, generated_sequence: list[int]) -> Tuple[Tuple[int, ...], int]:
        for seq_length in range(self.max_sequence_length, 0, -1):            
            if len(generated_sequence) < seq_length:
                continue
            candidate_sequence = tuple(generated_sequence[-seq_length:])
            if candidate_sequence in self.token_sequences:
                start_pos = len(generated_sequence) - seq_length
                return candidate_sequence, start_pos
        return None, -1