"""Textual TUI application for Prompt Weaver — Matrix Edition."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Footer, Static

from .engine import CombinatorialEngine
from .models import GeneratedPrompt
from .store import PromptStore
from .widgets import EntropyMeter, GlitchPrompt, HistoryLog, MatrixBanner

# ── neon category highlight colors ───────────────────────────────────────
CATEGORY_STYLES: dict[str, str] = {
    "subject_form": "bold bright_white",
    "material_substance": "bold bright_yellow",
    "texture_density": "bold bright_green",
    "light_behavior": "bold bright_cyan",
    "color_logic": "bold bright_magenta",
    "atmosphere_field": "bold #6688ff",
    "phenomenon_pattern": "bold #ff6644",
    "spatial_logic": "bold #aaffaa",
    "scale_perspective": "bold #ffaa44",
    "temporal_state": "bold #ff88ff",
    "setting_location": "bold #44ffcc",
    "medium_render": "bold #ff8866",
}

LABEL_STYLES: dict[str, str] = {
    "subject_form": "dim",
    "material_substance": "dim yellow",
    "texture_density": "dim green",
    "light_behavior": "dim cyan",
    "color_logic": "dim magenta",
    "atmosphere_field": "dim blue",
    "phenomenon_pattern": "dim red",
    "spatial_logic": "dim",
    "scale_perspective": "dim yellow",
    "temporal_state": "dim magenta",
    "setting_location": "dim cyan",
    "medium_render": "dim red",
}


def _highlight_prompt(prompt_text: str, components: dict[str, list[str]]) -> Text:
    """Highlight component words in a prompt string with category colors."""
    text = Text(prompt_text)
    for category, words in components.items():
        style = CATEGORY_STYLES.get(category, "bold bright_green")
        for word in words:
            start = 0
            while True:
                idx = prompt_text.find(word, start)
                if idx == -1:
                    break
                text.stylize(style, idx, idx + len(word))
                start = idx + len(word)
    return text


class PromptWeaverApp(App[None]):
    """Combinatorial prompt generator TUI — Matrix Edition."""

    TITLE = "promptweaver"
    SUB_TITLE = "// combinatorial prompt generator"

    CSS = """
    Screen {
        background: #000000;
        color: #00ff41;
    }

    MatrixBanner {
        dock: top;
        height: auto;
        background: #000000;
        padding: 0 2;
    }

    #body {
        height: 1fr;
    }

    #main {
        width: 3fr;
        margin: 0 1;
    }

    #sidebar {
        width: 1fr;
        min-width: 28;
        max-width: 38;
        margin: 0 1 0 0;
    }

    #negative-display, #components-display {
        margin-bottom: 1;
    }

    Footer {
        background: #001100;
        color: #00ff41;
    }
    """

    BINDINGS = [
        Binding("space", "next_prompt", "GENERATE", priority=True),
        Binding("enter", "next_prompt", "GENERATE", show=False, priority=True),
        Binding("t", "cycle_template", "TEMPLATE"),
        Binding("c", "copy_prompt", "COPY"),
        Binding("q", "quit_app", "EXIT"),
    ]

    def __init__(self, db_path: Optional[Path] = None) -> None:
        super().__init__()
        self.engine = CombinatorialEngine()
        self.store = PromptStore(db_path=db_path)
        self.current: Optional[GeneratedPrompt] = None
        self._template_filter: Optional[str] = None
        self._template_idx: int = 0

    def compose(self) -> ComposeResult:
        yield MatrixBanner()
        with Horizontal(id="body"):
            with VerticalScroll(id="main"):
                yield GlitchPrompt(id="prompt-display")
                yield Static(id="negative-display")
                yield Static(id="components-display")
                yield EntropyMeter(id="entropy-display")
            with Vertical(id="sidebar"):
                yield HistoryLog(id="history-log")
        yield Footer()

    def on_mount(self) -> None:
        self._generate()

    # ── generation ────────────────────────────────────────────────────

    def _generate(self) -> None:
        self.current = self.engine.generate_unique(
            self.store.seen_hashes,
            template_id=self._template_filter,
        )
        self.store.save(self.current)
        self._render()

    # ── rendering ─────────────────────────────────────────────────────

    def _render(self) -> None:
        if not self.current:
            return

        p = self.current

        # Positive prompt — glitch decode animation
        highlighted = _highlight_prompt(p.positive, p.components)
        self.query_one("#prompt-display", GlitchPrompt).decode(
            plain_text=p.positive,
            final_renderable=highlighted,
            title=f"[bold bright_green]{p.template_id}[/]",
            subtitle=f"[dim green]0x{p.hash}[/]",
        )

        # Negative prompt
        self.query_one("#negative-display", Static).update(
            Panel(
                Text(p.negative, style="dim italic #aa4444"),
                title="[dim red]// negative[/]",
                border_style="#664444",
                padding=(0, 2),
            )
        )

        # Component breakdown table
        tbl = Table(
            show_header=False, box=None, padding=(0, 2, 0, 0), expand=True
        )
        tbl.add_column("Category", min_width=22, no_wrap=True)
        tbl.add_column("Selection")
        for cat, words in p.components.items():
            tbl.add_row(
                Text(cat, style=LABEL_STYLES.get(cat, "dim green")),
                Text(", ".join(words), style=CATEGORY_STYLES.get(cat, "green")),
            )
        self.query_one("#components-display", Static).update(
            Panel(
                tbl,
                title="[dim green]// components[/]",
                border_style="#006600",
                padding=(0, 1),
            )
        )

        # Entropy meter
        self.query_one("#entropy-display", EntropyMeter).set_progress(
            count=self.store.count,
            total=self.engine.total_combinations,
            template_filter=self._template_filter,
        )

        # History log
        self.query_one("#history-log", HistoryLog).add_entry(
            hash_str=p.hash,
            template_id=p.template_id,
        )

    # ── actions ───────────────────────────────────────────────────────

    def action_next_prompt(self) -> None:
        self._generate()

    def action_cycle_template(self) -> None:
        options: list[Optional[str]] = [None, *self.engine.template_ids]
        self._template_idx = (self._template_idx + 1) % len(options)
        self._template_filter = options[self._template_idx]
        label = self._template_filter or "all"
        self.notify(f"template: {label}", timeout=2)
        self._generate()

    def action_copy_prompt(self) -> None:
        if not self.current:
            return
        text = self.current.positive.encode()
        for cmd in (
            ["wl-copy"],
            ["xclip", "-selection", "clipboard"],
            ["xsel", "--clipboard", "--input"],
        ):
            try:
                subprocess.run(
                    cmd, input=text, capture_output=True, timeout=2, check=True
                )
                self.notify("copied!", timeout=1)
                return
            except (
                FileNotFoundError,
                subprocess.TimeoutExpired,
                subprocess.CalledProcessError,
            ):
                continue
        self.notify("no clipboard tool found", severity="warning", timeout=2)

    def action_quit_app(self) -> None:
        self.store.close()
        self.exit()
