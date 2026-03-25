"""CLI entry point for Prompt Weaver."""

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="promptweaver",
        description="Combinatorial prompt generator that never repeats",
    )
    parser.add_argument(
        "--export",
        type=Path,
        metavar="FILE",
        help="Export all generated prompts to a JSON file and exit",
    )
    parser.add_argument(
        "--db",
        type=Path,
        metavar="PATH",
        help="Custom database path (default: ~/.local/share/promptweaver/prompts.db)",
    )
    args = parser.parse_args()

    if args.export:
        from .store import PromptStore

        store = PromptStore(db_path=args.db)
        prompts = store.get_all()
        with open(args.export, "w") as f:
            json.dump(
                [p.model_dump(mode="json") for p in prompts],
                f,
                indent=2,
                default=str,
            )
        print(f"Exported {len(prompts)} prompts to {args.export}")
        store.close()
        return

    try:
        from .app import PromptWeaverApp

        app = PromptWeaverApp(db_path=args.db)
        app.run()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
