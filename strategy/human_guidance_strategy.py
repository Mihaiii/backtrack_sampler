import torch
from typing import List
from .base_strategy import BaseStrategy
from ..provider.base_provider import BaseProvider
import curses


class HumanGuidanceStrategy(BaseStrategy):
    def __init__(
        self, provider: BaseProvider, top_k: int = 3, min_autopass: float = 1.0
    ):
        """
        top_k: The number of top tokens to consider for the human guidance menu.
        min_autopass: The minimum probability of the most probable token for the autopass to be triggered.
            Value is between 0 and 1, where 0 means the menu for manual selection is never shown and the
            top token will always be selected (same as temperature=0) and 1 means the menu is always shown.
        """
        self.provider = provider
        self.top_k = top_k
        self.min_autopass = min_autopass
        self.reset()

    def reset(self) -> None:
        self.keep_idx = 0
        self.go_back = False

    def get_keep_index(self) -> int:
        return self.keep_idx

    def on_logits(
        self, logits: torch.FloatTensor, continuation_tokens: List[int]
    ) -> torch.FloatTensor:
        return logits

    def on_probs(
        self,
        probs: torch.FloatTensor,
        continuation_tokens: List[int]
    ) -> torch.FloatTensor:
        top_k_probs, top_k_indices = torch.topk(probs, self.top_k)
        list_probs = top_k_probs.flatten().tolist()
        list_indices = top_k_indices.flatten().tolist()
        decoded_tokens = []
        for indices in list_indices:
            decoded_indices = self.provider.decode([indices])
            decoded_tokens.append(decoded_indices)
        options = list(zip(decoded_tokens, list_probs))
        if len(continuation_tokens) > 0:
            options = options + [("Go back", None)]
        generated_text = self.provider.decode(continuation_tokens)
        selected_option_index = 0
        if list_probs[0] < self.min_autopass:
            selected_option_index = curses.wrapper(self._menu, options, generated_text)

        # If the user selects the go back option, we'll backtrack by one token
        # Otherwise, we'll make the selected token 10000x more probable than it was before
        if selected_option_index >= self.top_k:
            self.go_back = True
        else:
            probs[:, list_indices[selected_option_index]] *= 10000
            probs = probs / probs.sum()
        return probs

    def on_next_token(
        self, continuation_tokens: List[int], probs: torch.FloatTensor
    ) -> None:
        self.keep_idx = len(continuation_tokens)

    def backtrack(self, continuation_tokens: List[int]) -> List[int]:
        if self.go_back:
            # We go back twice: once for the latest generated token,
            # whatever that might be, since we did not intervene,
            # and once the actual go back case
            continuation_tokens.pop()
            continuation_tokens.pop()
            self.go_back = False
            self.keep_idx = len(continuation_tokens)
        return continuation_tokens

    def _menu(self, stdscr, options, generated_text):
        curses.curs_set(0)  # Hide the cursor
        current_row = 0

        def get_last_row_of_text(stdscr, text, start_row, max_width):
            lines = []
            for line in text.splitlines():  # Preserve explicit new lines
                while len(line) > max_width:  # Wrap long lines
                    lines.append(line[:max_width])
                    line = line[max_width:]
                lines.append(line)

            # Calculate the last row where the text will be displayed
            start_options_row = start_row + len(lines) - 1
            return start_options_row

        height, width = stdscr.getmaxyx()
        start_row = 3
        max_width = width
        start_options_row = get_last_row_of_text(
            stdscr, generated_text, start_row, max_width
        )

        def get_display_text(option):
            if option[1] is None:  # Go back option
                return repr(option[0])
            return repr(f"{option[0]} ({option[1]*100:.2f}%)")

        def display_menu(stdscr, current_row):
            stdscr.clear()
            stdscr.addstr(1, 2, generated_text, curses.A_BOLD)

            for idx, row in enumerate(options):
                x = 2
                y = start_options_row + idx
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

            if key == curses.KEY_UP and current_row == 0:
                current_row = len(options) - 1
            elif key == curses.KEY_DOWN and current_row == len(options) - 1:
                current_row = 0
            elif key == curses.KEY_UP and current_row > 0:
                current_row -= 1
            elif key == curses.KEY_DOWN and current_row < len(options) - 1:
                current_row += 1
            elif key == ord("\n"):
                break  # User pressed Enter, exit the loop and resume execution

        return current_row
