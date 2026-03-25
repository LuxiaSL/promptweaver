"""Custom matrix-themed widgets for PromptWeaver TUI."""

from __future__ import annotations

import math
import random
from typing import Optional

from rich.console import RenderableType
from rich.panel import Panel
from rich.text import Text
from textual.timer import Timer
from textual.widgets import Static

try:
    import pyfiglet
except ImportError:
    pyfiglet = None  # type: ignore[assignment]

# Glitch character palette
GLITCH_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*<>{}[]|/\\~=+-"


class MatrixBanner(Static):
    """ASCII art title banner rendered with pyfiglet."""

    DEFAULT_CSS = """
    MatrixBanner {
        height: auto;
        padding: 0 2;
        color: #00ff41;
    }
    """

    def __init__(
        self,
        text: str = "PROMPTWEAVER",
        font: str = "doom",
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self._banner_text = text
        self._font = font

    def render(self) -> Text:
        if pyfiglet is not None:
            try:
                banner = pyfiglet.figlet_format(
                    self._banner_text, font=self._font, width=120
                )
            except Exception:
                banner = self._banner_text
        else:
            banner = f"  [ {self._banner_text} ]  "
        return Text(banner.rstrip("\n"), style="bold bright_green")


class GlitchPrompt(Static):
    """Prompt display with glitch-decode reveal animation.

    Characters scramble randomly then "lock in" left-to-right, mimicking
    a password cracker / matrix decode sequence.
    """

    DEFAULT_CSS = """
    GlitchPrompt {
        height: auto;
        margin-bottom: 1;
    }
    """

    DECODE_FPS: float = 1 / 24
    TOTAL_FRAMES: int = 14

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._target_plain: str = ""
        self._target_renderable: RenderableType = Text("")
        self._panel_title: str = ""
        self._panel_subtitle: str = ""
        self._frame: int = 0
        self._timer: Optional[Timer] = None

    def decode(
        self,
        plain_text: str,
        final_renderable: RenderableType,
        title: str = "",
        subtitle: str = "",
    ) -> None:
        """Start glitch-decode animation, ending with final_renderable."""
        self._target_plain = plain_text
        self._target_renderable = final_renderable
        self._panel_title = title
        self._panel_subtitle = subtitle
        self._frame = 0
        if self._timer is not None:
            self._timer.stop()
        self._timer = self.set_interval(self.DECODE_FPS, self._tick)

    def _tick(self) -> None:
        self._frame += 1

        if self._frame >= self.TOTAL_FRAMES:
            # Done — show final styled content with bright border flash
            self.update(
                Panel(
                    self._target_renderable,
                    title=self._panel_title,
                    subtitle=self._panel_subtitle,
                    border_style="bright_green",
                    padding=(1, 2),
                )
            )
            if self._timer is not None:
                self._timer.stop()
                self._timer = None
            return

        # Build partially decoded text with occasional color flashes
        progress = self._frame / self.TOTAL_FRAMES
        locked_count = int(len(self._target_plain) * progress)

        text = Text()
        for i, ch in enumerate(self._target_plain):
            if i < locked_count:
                text.append(ch, style="bright_green")
            elif ch in (" ", ",", ".", "\n", ";", ":"):
                text.append(ch)
            else:
                glyph = random.choice(GLITCH_CHARS)
                roll = random.random()
                if roll < 0.10:
                    text.append(glyph, style="bright_cyan")
                elif roll < 0.14:
                    text.append(glyph, style="bright_magenta")
                else:
                    text.append(glyph, style="green")

        border_color = random.choice(
            ["green", "bright_green", "#006600", "cyan"]
        )
        self.update(
            Panel(
                text,
                title=self._panel_title,
                subtitle="[dim green]decoding...[/]",
                border_style=border_color,
                padding=(1, 2),
            )
        )

    def set_static(
        self,
        renderable: RenderableType,
        title: str = "",
        subtitle: str = "",
    ) -> None:
        """Set content immediately without animation."""
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        self.update(
            Panel(
                renderable,
                title=title,
                subtitle=subtitle,
                border_style="bright_green",
                padding=(1, 2),
            )
        )


class HistoryLog(Static):
    """Scrolling log of recent prompt generations."""

    DEFAULT_CSS = """
    HistoryLog {
        height: 100%;
        width: 100%;
    }
    """

    MAX_ENTRIES: int = 50
    VISIBLE_ENTRIES: int = 18

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._entries: list[tuple[str, str]] = []  # (hash, template_id)

    def add_entry(self, hash_str: str, template_id: str) -> None:
        """Append a generation to the history log."""
        self._entries.append((hash_str, template_id))
        if len(self._entries) > self.MAX_ENTRIES:
            self._entries = self._entries[-self.MAX_ENTRIES:]
        self._refresh_display()

    def _refresh_display(self) -> None:
        visible = list(reversed(self._entries))[: self.VISIBLE_ENTRIES]
        text = Text()
        for i, (h, tid) in enumerate(visible):
            if i == 0:
                # Most recent — bright highlight
                text.append(" > ", style="bold bright_green")
                text.append(f"0x{h}", style="bold bright_green")
                text.append(f"\n   {tid}\n", style="green")
            else:
                style = "green" if i <= 3 else "dim green"
                text.append(f"   0x{h}", style=style)
                text.append(f"\n   {tid}\n", style="dim")

        self.update(
            Panel(
                text or Text("  awaiting generation...", style="dim green"),
                title="[dim green]// history[/]",
                border_style="#006600",
                padding=(0, 1),
            )
        )


class EntropyMeter(Static):
    """Visual meter showing exploration of the combinatorial space."""

    DEFAULT_CSS = """
    EntropyMeter {
        height: auto;
        margin-top: 1;
    }
    """

    BAR_WIDTH: int = 32

    def set_progress(
        self,
        count: int,
        total: int,
        template_filter: Optional[str] = None,
    ) -> None:
        """Update the entropy meter with current coverage stats."""
        pct = (count / total * 100) if total > 0 else 0.0

        # Log-scale bar — actual percentage is astronomically small,
        # so linear would be invisible. Log gives visual feedback early.
        if total > 1 and count > 0:
            log_progress = math.log10(count + 1) / math.log10(total)
        else:
            log_progress = 0.0

        filled = int(self.BAR_WIDTH * min(log_progress, 1.0))
        empty = self.BAR_WIDTH - filled

        bar = Text()
        bar.append("  ", style="")
        bar.append("\u2588" * filled, style="bright_green")
        bar.append("\u2591" * empty, style="#003300")
        bar.append("  ", style="")
        bar.append(f"#{count:,}", style="bold bright_green")
        bar.append("  of  ", style="dim")
        bar.append(f"~{total:,}", style="dim green")
        bar.append("  \u00b7  ", style="dim")
        bar.append(f"{pct:.8f}%", style="dim cyan")

        if template_filter:
            bar.append(f"  [{template_filter}]", style="cyan")
        else:
            bar.append("  [all]", style="dim")

        self.update(bar)
