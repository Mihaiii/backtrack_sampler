import torch
from typing import List
from .base_strategy import BaseStrategy
from ..provider.base_provider import BaseProvider
import curses

class HumanGuidanceStrategy(BaseStrategy):
    def __init__(self, provider: BaseProvider, top_k: int = 3):
        self.provider = provider
        self.top_k = top_k
        self.reset()

    def reset(self) -> None:
        self.keep_idx = 0
        self.go_back = False
        
    def get_keep_index(self) -> int:
        return self.keep_idx

    def on_logits(self, logits: torch.FloatTensor, continuation_tokens: List[int]) -> torch.FloatTensor:
        return logits

    def on_probs(self, probs: torch.FloatTensor, continuation_tokens: List[int]) -> torch.FloatTensor:
        top_k_probs, top_k_indices = torch.topk(probs, self.top_k)
        list_indices = top_k_indices.flatten().tolist()
        decoded_tokens = []
        for indices in list_indices:
            decoded_indices = self.provider.decode([indices])
            decoded_tokens.append(decoded_indices)
        result = list(zip(decoded_tokens, top_k_probs.flatten().tolist()))
        if len(continuation_tokens) > 0:
            result = result + [("Go back", None)]
        generated_text = self.provider.decode(continuation_tokens)
        result_index = curses.wrapper(self._menu, result, generated_text)
        # If the user selects the go back option, we'll backtrack by one token
        # Otherwise, we'll make the selected token 1000x more probable than it was before
        if result_index >= self.top_k:
            self.go_back = True
        else:
            probs[:, list_indices[result_index]] *= 10000
            probs = probs / probs.sum()
        return probs

    def on_next_token(self, continuation_tokens: List[int], probs: torch.FloatTensor) -> None:
        self.keep_idx = len(continuation_tokens)

    def backtrack(self, continuation_tokens: List[int]) -> List[int]:
        if self.go_back:
            #We go back twice: once for the latest generated token,
            #whatever that might be, since we did not intervene,
            #and once the actual go back case
            continuation_tokens.pop()
            continuation_tokens.pop()
            self.go_back = False
            self.keep_idx = len(continuation_tokens)
        return continuation_tokens
    
    def _menu(self, stdscr, options, title):
        curses.curs_set(0)  # Hide the cursor
        current_row = 0

        def get_display_text(option):
            if option[1] is None: # Go back option
                return option[0]
            return f"{option[0]} ({option[1]*100:.2f}%)"
                
        def display_menu(stdscr, current_row):
            stdscr.clear()
            stdscr.addstr(1, 2, title, curses.A_BOLD)
            
            for idx, row in enumerate(options):
                x = 2
                y = 3 + idx
                if idx == current_row:
                    stdscr.attron(curses.color_pair(1))  # Highlight selected row
                    stdscr.addstr(y, x, get_display_text(row))
                    stdscr.attroff(curses.color_pair(1))
                else:
                    stdscr.addstr(y, x, get_display_text(row))
            stdscr.refresh()

        curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)
        
        while True:
            display_menu(stdscr, current_row)
            key = stdscr.getch()

            if key == curses.KEY_UP and current_row > 0:
                current_row -= 1
            elif key == curses.KEY_DOWN and current_row < len(options) - 1:
                current_row += 1
            elif key == ord('\n'):
                break  # User pressed Enter, exit the loop and resume execution

        return current_row
