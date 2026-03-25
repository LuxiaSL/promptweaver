"""SQLite persistence for generated prompts."""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

from .models import GeneratedPrompt

logger = logging.getLogger(__name__)


class PromptStore:
    """Persists generated prompts and tracks seen hashes for deduplication."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        if db_path is None:
            db_path = Path.home() / ".local" / "share" / "promptweaver" / "prompts.db"

        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        self._hash_cache: set[str] = self._load_hashes()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hash TEXT UNIQUE NOT NULL,
                template_id TEXT NOT NULL,
                positive TEXT NOT NULL,
                negative TEXT NOT NULL,
                components TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_hash ON prompts(hash);
            CREATE INDEX IF NOT EXISTS idx_created ON prompts(created_at DESC);
        """
        )
        self._conn.commit()

    def _load_hashes(self) -> set[str]:
        cur = self._conn.execute("SELECT hash FROM prompts")
        return {row["hash"] for row in cur.fetchall()}

    def save(self, prompt: GeneratedPrompt) -> None:
        """Save a prompt (skips silently if hash already exists)."""
        try:
            self._conn.execute(
                """INSERT OR IGNORE INTO prompts
                   (hash, template_id, positive, negative, components, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    prompt.hash,
                    prompt.template_id,
                    prompt.positive,
                    prompt.negative,
                    json.dumps(prompt.components),
                    prompt.created_at.isoformat(),
                ),
            )
            self._conn.commit()
            self._hash_cache.add(prompt.hash)
        except sqlite3.Error as e:
            logger.warning(f"Failed to save prompt: {e}")

    @property
    def seen_hashes(self) -> set[str]:
        return self._hash_cache

    @property
    def count(self) -> int:
        return len(self._hash_cache)

    def get_recent(self, limit: int = 50) -> list[GeneratedPrompt]:
        cur = self._conn.execute(
            "SELECT * FROM prompts ORDER BY id DESC LIMIT ?", (limit,)
        )
        return [self._row_to_prompt(row) for row in cur.fetchall()]

    def get_all(self) -> list[GeneratedPrompt]:
        cur = self._conn.execute("SELECT * FROM prompts ORDER BY id ASC")
        return [self._row_to_prompt(row) for row in cur.fetchall()]

    @staticmethod
    def _row_to_prompt(row: sqlite3.Row) -> GeneratedPrompt:
        return GeneratedPrompt(
            hash=row["hash"],
            template_id=row["template_id"],
            positive=row["positive"],
            negative=row["negative"],
            components=json.loads(row["components"]),
            created_at=row["created_at"],
        )

    def close(self) -> None:
        try:
            self._conn.close()
        except sqlite3.Error:
            pass
