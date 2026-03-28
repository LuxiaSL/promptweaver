#!/usr/bin/env python3
"""
Diversity Selection Pipeline — Gated Dual-Space Logic
═══════════════════════════════════════════════════════════════════════════════

Uses logic gates instead of alpha-blending to combine two orthogonal
embedding spaces (visual: OpenCLIP/CLIP, semantic: T5).

Gate logic per pipeline stage:
    Contamination:  AND  — must be closest to correct centroid in BOTH spaces
    Redundancy:     OR   — similar in EITHER space → flag
    Selection:      OR-distance — far in EITHER space → keep both
    Opposites:      T5-only — semantic distance for negative prompts

Usage:
    uv run python -m apeiron.tools.select \
        -i data/generated_candidates_20260325_193614.yaml \
        --k 60 --clip-model openclip --t5-model google/t5-v1_1-xxl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .embeddings import (
    DualEmbeddings,
    DualSpaceEmbedder,
    DiversityStats,
    analyze_diversity,
    farthest_point_sampling,
    greedy_opposite_pairs,
    load_components_yaml,
)

logger = logging.getLogger(__name__)

NEGATABLE_CATEGORIES = frozenset({
    "color_logic", "light_behavior", "atmosphere_field",
    "temporal_state", "texture_density", "medium_render",
})


# ═══════════════════════════════════════════════════════════════════════════════
# EMBEDDING CACHE
# ═══════════════════════════════════════════════════════════════════════════════


class EmbeddingCache:
    """Compute once, lookup forever."""

    def __init__(self, embedder: DualSpaceEmbedder, all_words: list[str]) -> None:
        unique = list(dict.fromkeys(all_words))
        logger.info(f"Computing embeddings for {len(unique)} unique words (one-shot)...")

        clip_emb = embedder.clip_embedder.encode_batch(unique)
        t5_emb: np.ndarray | None = None
        if embedder.t5_embedder:
            t5_emb = embedder.t5_embedder.encode_batch(unique)

        self._clip = {w: clip_emb[i] for i, w in enumerate(unique)}
        self._t5 = {w: t5_emb[i] for i, w in enumerate(unique)} if t5_emb is not None else None
        self.has_t5 = t5_emb is not None

        logger.info(f"Cache ready: {len(unique)} words, visual={clip_emb.shape[1]}d"
                     + (f", semantic={t5_emb.shape[1]}d" if t5_emb is not None else ""))

    def get(self, words: list[str]) -> DualEmbeddings:
        clip = np.array([self._clip[w] for w in words])
        t5 = np.array([self._t5[w] for w in words]) if self._t5 else None
        return DualEmbeddings(words=words, clip=clip, t5=t5)


# ═══════════════════════════════════════════════════════════════════════════════
# GATED SIMILARITY MATRICES
# ═══════════════════════════════════════════════════════════════════════════════


def sim_redundancy_or(emb: DualEmbeddings) -> np.ndarray:
    """
    OR gate for redundancy: max(clip_sim, t5_sim) per pair.
    If EITHER model says similar, flag it. Catches both visual and
    semantic duplicates.
    """
    clip_sim = emb.clip @ emb.clip.T
    if emb.t5 is None:
        return clip_sim
    t5_sim = emb.t5 @ emb.t5.T
    return np.maximum(clip_sim, t5_sim)


def sim_selection_or_distance(emb: DualEmbeddings) -> np.ndarray:
    """
    OR-distance for selection: min(clip_sim, t5_sim) per pair.
    If EITHER model says they're different, treat them as different.
    Farthest-point sampling on this matrix keeps items that are distinct
    in at least one dimension.
    """
    clip_sim = emb.clip @ emb.clip.T
    if emb.t5 is None:
        return clip_sim
    t5_sim = emb.t5 @ emb.t5.T
    return np.minimum(clip_sim, t5_sim)


def sim_t5_only(emb: DualEmbeddings) -> np.ndarray:
    """T5-only similarity for opposite pairing (semantic distance)."""
    if emb.t5 is not None:
        return emb.t5 @ emb.t5.T
    return emb.clip @ emb.clip.T


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: CONTAMINATION — AND GATE
# ═══════════════════════════════════════════════════════════════════════════════


def compute_centroids(
    categories: dict[str, list[str]],
    cache: EmbeddingCache,
) -> dict[str, tuple[np.ndarray, np.ndarray | None]]:
    centroids: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
    for cat, words in categories.items():
        if not words:
            continue
        emb = cache.get(words)
        clip_c = emb.clip.mean(axis=0)
        clip_c /= np.linalg.norm(clip_c)
        t5_c = None
        if emb.t5 is not None:
            t5_c = emb.t5.mean(axis=0)
            t5_c /= np.linalg.norm(t5_c)
        centroids[cat] = (clip_c, t5_c)
    return centroids


def reallocate_contaminated_and(
    categories: dict[str, list[str]],
    cache: EmbeddingCache,
    min_delta: float = 0.01,
) -> tuple[dict[str, list[str]], int]:
    """
    AND-gated contamination: a word is misclassified only if it's closer
    to the WRONG centroid in BOTH embedding spaces. This is a much stronger
    signal than alpha-blended similarity — it means the word looks wrong
    AND means the wrong thing.
    """
    centroids = compute_centroids(categories, cache)
    result = {cat: list(words) for cat, words in categories.items()}
    moves: list[tuple[str, str, str, float, float]] = []

    for cat, words in categories.items():
        emb = cache.get(words)

        for i, word in enumerate(words):
            # Compute similarity to each centroid in BOTH spaces
            clip_sims: dict[str, float] = {}
            t5_sims: dict[str, float] = {}

            for other_cat, (clip_c, t5_c) in centroids.items():
                clip_sims[other_cat] = float(np.dot(emb.clip[i], clip_c))
                if t5_c is not None and emb.t5 is not None:
                    t5_sims[other_cat] = float(np.dot(emb.t5[i], t5_c))

            # AND gate: misclassified only if BOTH spaces agree it's wrong
            clip_closest = max(clip_sims, key=lambda c: clip_sims[c])
            clip_delta = clip_sims.get(clip_closest, 0) - clip_sims.get(cat, 0)

            if t5_sims:
                t5_closest = max(t5_sims, key=lambda c: t5_sims[c])
                t5_delta = t5_sims.get(t5_closest, 0) - t5_sims.get(cat, 0)

                # Both spaces must agree: wrong category AND same target
                if (clip_closest != cat and t5_closest != cat
                        and clip_closest == t5_closest  # Both point to same target
                        and clip_delta > min_delta and t5_delta > min_delta):
                    moves.append((word, cat, clip_closest, clip_delta, t5_delta))
            else:
                # Fallback: single-space check
                if clip_closest != cat and clip_delta > min_delta:
                    moves.append((word, cat, clip_closest, clip_delta, 0.0))

    # Apply moves
    moved = 0
    for word, from_cat, to_cat, cd, td in moves:
        if word in result[from_cat] and word not in result.get(to_cat, []):
            result[from_cat].remove(word)
            result.setdefault(to_cat, []).append(word)
            moved += 1

    if moved > 0:
        logger.info(f"  AND-gate reallocation: moved {moved} items (both spaces agree)")
        top = sorted(moves, key=lambda x: x[3] + x[4], reverse=True)[:5]
        for word, frm, to, cd, td in top:
            logger.info(f"    '{word}': {frm} → {to} (clip={cd:+.3f}, t5={td:+.3f})")
        if len(moves) > 5:
            logger.info(f"    ... and {len(moves) - 5} more")

    return result, moved


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: PER-CATEGORY SELECTION
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SelectionResult:
    category: str
    candidates_count: int
    filtered_count: int
    selected_count: int
    selected_words: list[str]
    opposites: dict[str, str]
    before_stats: DiversityStats
    after_stats: DiversityStats
    elbow_k: int | None
    redundant_dropped: int


def compute_elbow(sim_matrix: np.ndarray, max_k: int | None = None) -> int:
    n = sim_matrix.shape[0]
    if n < 4:
        return n
    max_k = max_k or n

    masked = sim_matrix.copy()
    np.fill_diagonal(masked, np.inf)
    min_idx = np.unravel_index(np.argmin(masked), masked.shape)
    selected = [min_idx[0], min_idx[1]]
    selected_set = set(selected)

    min_sim = np.minimum(sim_matrix[:, selected[0]], sim_matrix[:, selected[1]])
    distances: list[float] = []

    while len(selected) < min(max_k, n):
        sim_masked = min_sim.copy()
        for idx in selected_set:
            sim_masked[idx] = np.inf
        best = int(np.argmin(sim_masked))
        if best in selected_set:
            break
        distances.append(float(1.0 - sim_masked[best]))
        selected.append(best)
        selected_set.add(best)
        min_sim = np.minimum(min_sim, sim_matrix[:, best])

    if len(distances) < 3:
        return len(distances) + 2

    diffs = [distances[i] - distances[i + 1] for i in range(len(distances) - 1)]
    return int(np.argmax(diffs)) + 3


def select_category(
    category: str,
    words: list[str],
    cache: EmbeddingCache,
    k: int,
    redundancy_threshold: float,
) -> SelectionResult:
    """
    Gated selection pipeline:
        1. OR-gate redundancy filter (similar in either space → drop)
        2. OR-distance farthest-point selection (different in either space → keep)
        3. T5-only opposite pairing (semantic distance)
    """
    logger.info(f"\n{'─'*60}")
    logger.info(f"{category} ({len(words)} candidates)")
    logger.info(f"{'─'*60}")

    if not words:
        logger.warning(f"  Empty category!")
        empty_stats = DiversityStats(0, 0, 0, 0, 0, ("", "", 0), ("", "", 0), 0, redundancy_threshold)
        return SelectionResult(category, 0, 0, 0, [], {}, empty_stats, empty_stats, None, 0)

    emb = cache.get(words)

    # Before stats (using OR-distance for reporting consistency)
    sim_before = sim_selection_or_distance(emb)
    before_stats = analyze_diversity(sim_before, words, redundancy_threshold)

    # ── REDUNDANCY: OR gate ─────────────────────────────────────────
    # Flag pairs similar in EITHER space
    sim_or = sim_redundancy_or(emb)
    n = len(words)
    drop: set[int] = set()
    for i in range(n):
        if i in drop:
            continue
        for j in range(i + 1, n):
            if j in drop:
                continue
            if sim_or[i, j] > redundancy_threshold:
                drop.add(j)

    kept = [w for i, w in enumerate(words) if i not in drop]
    num_dropped = len(drop)
    if num_dropped:
        logger.info(f"  OR-gate redundancy (>{redundancy_threshold}): dropped {num_dropped}")
    logger.info(f"  After filter: {len(kept)} candidates")

    if not kept:
        logger.warning(f"  No candidates remain!")
        return SelectionResult(category, len(words), 0, 0, [], {},
                               before_stats, before_stats, None, num_dropped)

    # ── SELECTION: OR-distance ──────────────────────────────────────
    # Farthest-point in min(clip_sim, t5_sim) space
    # Items different in EITHER dimension are treated as different
    emb_kept = cache.get(kept)
    sim_sel = sim_selection_or_distance(emb_kept)

    elbow_k = compute_elbow(sim_sel, max_k=min(k + 20, len(kept)))
    effective_k = min(k, len(kept))
    logger.info(f"  Elbow at k={elbow_k} (using k={effective_k})")

    selected_idx = farthest_point_sampling(sim_sel, effective_k)
    selected_words = [kept[i] for i in selected_idx]

    # After stats
    emb_sel = cache.get(selected_words)
    sim_after = sim_selection_or_distance(emb_sel)
    after_stats = analyze_diversity(sim_after, selected_words, redundancy_threshold)

    improvement = before_stats.mean_similarity - after_stats.mean_similarity
    logger.info(
        f"  Diversity: {before_stats.mean_similarity:.3f} → {after_stats.mean_similarity:.3f} "
        f"({improvement:+.3f})"
    )
    logger.info(
        f"  Closest pair: '{after_stats.most_similar_pair[0]}' <-> "
        f"'{after_stats.most_similar_pair[1]}' ({after_stats.most_similar_pair[2]:.3f})"
    )

    # ── OPPOSITES: T5-only ──────────────────────────────────────────
    opposites: dict[str, str] = {}
    if category in NEGATABLE_CATEGORIES:
        sim_t5 = sim_t5_only(emb_sel)
        opposites = greedy_opposite_pairs(sim_t5, selected_words)
        logger.info(f"  Computed {len(opposites)} opposite pairs (T5-only semantic distance)")

    return SelectionResult(
        category=category,
        candidates_count=len(words),
        filtered_count=len(kept),
        selected_count=len(selected_words),
        selected_words=selected_words,
        opposites=opposites,
        before_stats=before_stats,
        after_stats=after_stats,
        elbow_k=elbow_k,
        redundant_dropped=num_dropped,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════


def build_components_yaml(results: dict[str, SelectionResult]) -> dict[str, Any]:
    components: dict[str, list[dict[str, Any]]] = {}
    for cat, r in sorted(results.items()):
        items: list[dict[str, Any]] = []
        for word in r.selected_words:
            entry: dict[str, Any] = {"word": word}
            if word in r.opposites:
                entry["opposite"] = r.opposites[word]
            items.append(entry)
        components[cat] = items
    return {"components": components}


def build_analysis_yaml(results: dict[str, SelectionResult]) -> dict[str, Any]:
    analysis: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "gate_logic": {
            "contamination": "AND (both spaces must agree on misclassification)",
            "redundancy": "OR (similar in either space → drop)",
            "selection": "OR-distance (different in either space → keep)",
            "opposites": "T5-only (semantic distance)",
        },
        "categories": {},
    }
    for cat, r in sorted(results.items()):
        analysis["categories"][cat] = {
            "candidates": r.candidates_count,
            "after_filter": r.filtered_count,
            "selected": r.selected_count,
            "elbow_k": r.elbow_k,
            "redundant_dropped": r.redundant_dropped,
            "before_mean_sim": round(r.before_stats.mean_similarity, 4),
            "after_mean_sim": round(r.after_stats.mean_similarity, 4),
            "improvement": round(
                r.before_stats.mean_similarity - r.after_stats.mean_similarity, 4
            ),
            "most_similar_pair": {
                "words": list(r.after_stats.most_similar_pair[:2]),
                "similarity": round(r.after_stats.most_similar_pair[2], 4),
            },
        }
    return analysis


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select most diverse components using gated dual-space logic",
    )
    parser.add_argument("--input", "-i", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path)
    parser.add_argument("--k", type=int, default=60)
    parser.add_argument("--redundancy-threshold", type=float, default=0.85,
                        help="OR-gate threshold: flag if similar in either space (default: 0.85)")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32",
                        help="Visual encoder (use 'openclip' or 'ViT-bigG-14' for OpenCLIP)")
    parser.add_argument("--t5-model", default="google/flan-t5-large",
                        help="Semantic encoder (use google/t5-v1_1-xxl for diffusion T5)")
    parser.add_argument("--clip-only", action="store_true")
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--skip-contamination", action="store_true")
    parser.add_argument("--save-reallocated", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S",
    )

    if not args.input.exists():
        logger.error(f"Input not found: {args.input}")
        sys.exit(1)

    all_candidates = load_components_yaml(args.input)
    if not all_candidates:
        logger.error("No categories found")
        sys.exit(1)

    select_cats = set(all_candidates.keys())
    if args.categories:
        select_cats = {c for c in args.categories if c in all_candidates}
        if not select_cats:
            logger.error(f"No matching categories: {args.categories}")
            sys.exit(1)

    t5_model = None if args.clip_only else args.t5_model
    total = sum(len(v) for v in all_candidates.values())

    print(f"{'='*60}")
    print("GATED DUAL-SPACE SELECTION")
    print(f"{'='*60}")
    print(f"  Input:      {args.input}")
    print(f"  Categories: {len(all_candidates)} ({total} words)")
    print(f"  Selecting:  {len(select_cats)}")
    print(f"  Target k:   {args.k}")
    print(f"  Visual:     {args.clip_model}")
    print(f"  Semantic:   {t5_model or '(disabled)'}")
    print(f"  Gates:      contamination=AND, redundancy=OR, selection=OR-dist, opposites=T5")

    # ── EMBED ONCE ──────────────────────────────────────────────────
    embedder = DualSpaceEmbedder(clip_model=args.clip_model, t5_model=t5_model)
    all_words = [w for words in all_candidates.values() for w in words]
    cache = EmbeddingCache(embedder, all_words)

    # ── AND-GATE CONTAMINATION ──────────────────────────────────────
    if args.skip_contamination:
        working = {cat: list(words) for cat, words in all_candidates.items()}
        logger.info("Skipping contamination")
    else:
        working, num_moved = reallocate_contaminated_and(all_candidates, cache)

    if args.save_reallocated:
        with open(args.save_reallocated, "w") as f:
            yaml.dump({"components": {c: sorted(w) for c, w in working.items()}},
                      f, default_flow_style=False, allow_unicode=True, width=120)
        print(f"\nReallocated pools: {args.save_reallocated}")

    # ── PER-CATEGORY GATED SELECTION ────────────────────────────────
    results: dict[str, SelectionResult] = {}
    for cat in sorted(select_cats):
        results[cat] = select_category(
            category=cat,
            words=working[cat],
            cache=cache,
            k=args.k,
            redundancy_threshold=args.redundancy_threshold,
        )

    # ── SUMMARY ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SELECTION SUMMARY")
    print(f"{'='*60}")
    print(
        f"{'Category':<25} {'Cand':>5} {'Filt':>5} {'Sel':>4} "
        f"{'Elbow':>5} {'Before':>7} {'After':>7} {'Improv':>7}"
    )
    print("-" * 72)

    for cat, r in sorted(results.items()):
        imp = r.before_stats.mean_similarity - r.after_stats.mean_similarity
        print(
            f"{cat:<25} {r.candidates_count:>5} {r.filtered_count:>5} "
            f"{r.selected_count:>4} {r.elbow_k or '-':>5} "
            f"{r.before_stats.mean_similarity:>7.3f} "
            f"{r.after_stats.mean_similarity:>7.3f} {imp:>+7.3f}"
        )

    total_sel = sum(r.selected_count for r in results.values())
    sel_cands = sum(r.candidates_count for r in results.values())
    print(f"\nTotal selected: {total_sel} from {sel_cands} candidates")

    if args.dry_run:
        print("\n[DRY RUN]")
        return

    data_dir = Path(__file__).parent.parent / "data"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    comp_path = args.output or data_dir / f"selected_gated_{ts}.yaml"
    with open(comp_path, "w") as f:
        yaml.dump(build_components_yaml(results), f,
                  default_flow_style=False, allow_unicode=True, width=120)
    print(f"\nComponents: {comp_path}")

    analysis_path = comp_path.with_name(comp_path.stem + "_analysis" + comp_path.suffix)
    with open(analysis_path, "w") as f:
        yaml.dump(build_analysis_yaml(results), f,
                  default_flow_style=False, allow_unicode=True, width=120)
    print(f"Analysis:   {analysis_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
