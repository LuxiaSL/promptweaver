"""Textual TUI application for Prompt Weaver."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Footer, Header, Static

from .engine import CombinatorialEngine
from .models import GeneratedPrompt
from .store import PromptStore

# Category highlight colors (work on both light + dark terminals)
CATEGORY_STYLES: dict[str, str] = {
    "subject_form": "bold",
    "material_substance": "bold yellow",
    "texture_density": "bold green",
    "light_behavior": "bold cyan",
    "color_logic": "bold magenta",
    "atmosphere_field": "bold blue",
    "phenomenon_pattern": "bold red",
    "spatial_logic": "bold white",
    "scale_perspective": "bold yellow",
    "temporal_state": "bold green",
    "setting_location": "bold cyan",
    "medium_render": "bold magenta",
}

LABEL_STYLES: dict[str, str] = {
    "subject_form": "dim bold",
    "material_substance": "dim yellow",
    "texture_density": "dim green",
    "light_behavior": "dim cyan",
    "color_logic": "dim magenta",
    "atmosphere_field": "dim blue",
    "phenomenon_pattern": "dim red",
    "spatial_logic": "dim",
    "scale_perspective": "dim yellow",
    "temporal_state": "dim green",
    "setting_location": "dim cyan",
    "medium_render": "dim magenta",
}


def _highlight_prompt(prompt_text: str, components: dict[str, list[str]]) -> Text:
    """Highlight component words in a prompt string with category colors."""
    text = Text(prompt_text)
    for category, words in components.items():
        style = CATEGORY_STYLES.get(category, "bold")
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
    """Combinatorial prompt generator TUI."""

    TITLE = "prompt weaver"
    SUB_TITLE = "combinatorial prompt generator"

    CSS = """
    Screen {
        background: $surface;
    }
    #main {
        margin: 1 3;
    }
    #prompt-display, #negative-display, #components-display {
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        Binding("space", "next_prompt", "Next", priority=True),
        Binding("enter", "next_prompt", "Next", show=False, priority=True),
        Binding("t", "cycle_template", "Template"),
        Binding("c", "copy_prompt", "Copy"),
        Binding("q", "quit_app", "Quit"),
    ]

    def __init__(self, db_path: Optional[Path] = None) -> None:
        super().__init__()
        self.engine = CombinatorialEngine()
        self.store = PromptStore(db_path=db_path)
        self.current: Optional[GeneratedPrompt] = None
        self._template_filter: Optional[str] = None
        self._template_idx: int = 0

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with VerticalScroll(id="main"):
            yield Static(id="prompt-display")
            yield Static(id="negative-display")
            yield Static(id="components-display")
            yield Static(id="stats-display")
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

        # Positive prompt — highlighted components
        highlighted = _highlight_prompt(p.positive, p.components)
        self.query_one("#prompt-display", Static).update(
            Panel(
                highlighted,
                title=f"[bold cyan]{p.template_id}[/]",
                subtitle=f"[dim]{p.hash}[/]",
                border_style="cyan",
                padding=(1, 2),
            )
        )

        # Negative prompt
        self.query_one("#negative-display", Static).update(
            Panel(
                Text(p.negative, style="dim italic"),
                title="[dim]negative[/]",
                border_style="bright_black",
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
                Text(cat, style=LABEL_STYLES.get(cat, "dim")),
                Text(", ".join(words), style=CATEGORY_STYLES.get(cat, "")),
            )
        self.query_one("#components-display", Static).update(
            Panel(
                tbl,
                title="[dim]components[/]",
                border_style="bright_black",
                padding=(0, 1),
            )
        )

        # Stats line
        total = self.engine.total_combinations
        count = self.store.count
        pct = (count / total * 100) if total > 0 else 0.0
        filter_label = (
            f"  [cyan]{self._template_filter}[/]"
            if self._template_filter
            else "  [dim]all templates[/]"
        )
        self.query_one("#stats-display", Static).update(
            Text.from_markup(
                f"  [dim]#{count:,}[/]  of  [dim]~{total:,} possible[/]"
                f"  ·  [dim]{pct:.8f}%[/]{filter_label}"
            )
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
            except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
                continue
        self.notify("no clipboard tool found", severity="warning", timeout=2)

    def action_quit_app(self) -> None:
        self.store.close()
        self.exit()
