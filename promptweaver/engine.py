"""Combinatorial prompt generation engine.

Generates unique prompts by combining templates with component pools.
Each template defines a sentence structure with {category} placeholders.
Components are randomly selected from pools of 25-30 items per category.
The resulting combinatorial space is billions of unique prompts.
"""

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Optional

import yaml

from .models import Component, GeneratedPrompt, SlotSpec, Template


class CombinatorialEngine:
    """Generates unique prompts through template + component combination."""

    NEGATABLE_CATEGORIES = frozenset(
        {
            "color_logic",
            "light_behavior",
            "atmosphere_field",
            "temporal_state",
            "texture_density",
            "medium_render",
        }
    )

    BASE_NEGATIVES = ("low quality", "blurry", "text", "watermark")

    def __init__(
        self,
        templates_path: Optional[Path] = None,
        components_path: Optional[Path] = None,
    ) -> None:
        data_dir = Path(__file__).parent / "data"
        self._templates_path = templates_path or data_dir / "templates.yaml"
        self._components_path = components_path or data_dir / "components.yaml"

        self.templates: dict[str, Template] = {}
        self.components: dict[str, list[Component]] = {}
        self._slots_cache: dict[str, list[SlotSpec]] = {}

        self._load_templates()
        self._load_components()

    # ── loaders ───────────────────────────────────────────────────────

    def _load_templates(self) -> None:
        if not self._templates_path.exists():
            raise FileNotFoundError(f"Templates not found: {self._templates_path}")

        with open(self._templates_path) as f:
            data = yaml.safe_load(f)

        for raw in data.get("templates", []):
            t = Template(
                id=raw["id"],
                structure=raw["structure"],
                required_components=raw["required_components"],
                notes=raw.get("notes", ""),
            )
            self.templates[t.id] = t
            self._slots_cache[t.id] = self._parse_slots(t.structure)

    def _load_components(self) -> None:
        if not self._components_path.exists():
            raise FileNotFoundError(f"Components not found: {self._components_path}")

        with open(self._components_path) as f:
            data = yaml.safe_load(f)

        for category, items in data.get("components", {}).items():
            self.components[category] = [
                Component(word=item["word"], opposite=item.get("opposite"))
                for item in items
            ]

    @staticmethod
    def _parse_slots(structure: str) -> list[SlotSpec]:
        """Parse {category}, {category:N}, {category:N:sep} slot syntax."""
        pattern = r"\{(\w+)(?::(\d+))?(?::([^}]*))?\}"
        return [
            SlotSpec(
                category=m.group(1),
                count=int(m.group(2)) if m.group(2) else 1,
                separator=m.group(3) if m.group(3) is not None else " ",
            )
            for m in re.finditer(pattern, structure)
        ]

    # ── generation ────────────────────────────────────────────────────

    def generate(self, template_id: Optional[str] = None) -> GeneratedPrompt:
        """Generate a single random prompt."""
        template = (
            self.templates[template_id]
            if template_id and template_id in self.templates
            else random.choice(list(self.templates.values()))
        )
        slots = self._slots_cache[template.id]

        # Compute total needed per category across all slots
        needs: dict[str, int] = {}
        for s in slots:
            needs[s.category] = needs.get(s.category, 0) + s.count

        # Sample without replacement per category
        selections: dict[str, list[Component]] = {}
        for category, total in needs.items():
            pool = self.components.get(category, [])
            if not pool:
                continue
            selections[category] = random.sample(pool, min(total, len(pool)))

        # Fill template string
        positive = template.structure
        consumed: dict[str, int] = {}

        for slot in slots:
            idx = consumed.get(slot.category, 0)
            chosen = selections.get(slot.category, [])
            batch = chosen[idx : idx + slot.count]
            consumed[slot.category] = idx + slot.count

            replacement = (
                slot.separator.join(c.word for c in batch)
                if batch
                else f"[missing {slot.category}]"
            )

            if slot.count > 1:
                pat = (
                    f"{{{slot.category}:{slot.count}:{slot.separator}}}"
                    if slot.separator != " "
                    else f"{{{slot.category}:{slot.count}}}"
                )
            else:
                pat = f"{{{slot.category}}}"

            positive = positive.replace(pat, replacement, 1)

        # Build negative from semantic opposites
        opposites = [
            comp.opposite
            for cat in self.NEGATABLE_CATEGORIES
            for comp in selections.get(cat, [])
            if comp.opposite
        ]
        negative = ", ".join(opposites + list(self.BASE_NEGATIVES))

        # Canonical components dict
        comp_dict = {
            cat: [c.word for c in comps] for cat, comps in selections.items()
        }

        # Uniqueness hash (template + sorted components)
        canon = json.dumps(
            {"t": template.id, "c": {k: sorted(v) for k, v in comp_dict.items()}},
            sort_keys=True,
        )
        prompt_hash = hashlib.sha256(canon.encode()).hexdigest()[:16]

        return GeneratedPrompt(
            hash=prompt_hash,
            template_id=template.id,
            positive=positive,
            negative=negative,
            components=comp_dict,
        )

    def generate_unique(
        self,
        seen: set[str],
        template_id: Optional[str] = None,
        max_attempts: int = 100,
    ) -> GeneratedPrompt:
        """Generate a prompt whose hash is not in *seen*."""
        prompt = self.generate(template_id)
        for _ in range(max_attempts):
            if prompt.hash not in seen:
                return prompt
            prompt = self.generate(template_id)
        return prompt  # Astronomically unlikely with billions of combos

    # ── stats ─────────────────────────────────────────────────────────

    @property
    def total_combinations(self) -> int:
        """Estimated total unique combinations across all templates."""
        total = 0
        for tid, slots in self._slots_cache.items():
            cat_needs: dict[str, int] = {}
            for s in slots:
                cat_needs[s.category] = cat_needs.get(s.category, 0) + s.count
            product = 1
            for cat, n in cat_needs.items():
                pool = len(self.components.get(cat, []))
                perm = 1
                for i in range(n):
                    perm *= max(1, pool - i)
                product *= perm
            total += product
        return total

    @property
    def template_ids(self) -> list[str]:
        return list(self.templates.keys())
