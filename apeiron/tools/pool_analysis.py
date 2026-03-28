#!/usr/bin/env python3
"""
Cross-Pool Analysis & Dual-Space Confidence Scoring
═══════════════════════════════════════════════════════════════════════════════

Comprehensive analysis across multiple component pools:

Phase 1 — Set Analysis (no models needed):
  Cross-pool overlap: shared, unique, pairwise intersections
  Per-category breakdown with Jaccard similarity

Phase 2 — Dual-Space Confidence Scoring (OpenCLIP + T5-XXL):
  Per-word confidence = cosine(word, category_centroid) in each space
  Quadrant analysis: rock-solid, visual-anchor, semantic-anchor, weak
  Compare confidence distributions across pools

Phase 3 — Inter-Category Gap Analysis:
  Centroid-to-centroid distance matrix in both spaces
  Category proximity disagreements between spaces
  Closest pairs (merge candidates) and largest gaps (new-axis candidates)

Usage:
    # Set analysis only (instant, no models)
    uv run python -m apeiron.tools.pool_analysis --set-only

    # Full analysis with OpenCLIP + T5-XXL
    uv run python -m apeiron.tools.pool_analysis

    # Save full report
    uv run python -m apeiron.tools.pool_analysis -o data/pool_analysis_report.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .embeddings import (
    DualEmbeddings,
    DualSpaceEmbedder,
    load_components_yaml,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# POOL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

DATA_DIR = Path(__file__).parent.parent / "data"

POOL_PATHS: dict[str, Path] = {
    "original": DATA_DIR / "components.yaml",
    "curated_v2": DATA_DIR / "curated_v2.yaml",
    "gated": DATA_DIR / "selected_gated_openclip_t5xxl.yaml",
}


@dataclass
class WordConfidence:
    """Per-word confidence scores in both embedding spaces."""

    word: str
    category: str
    clip_confidence: float  # cosine to own category centroid in CLIP
    t5_confidence: float  # cosine to own category centroid in T5
    clip_best_category: str  # closest centroid in CLIP
    t5_best_category: str  # closest centroid in T5
    clip_best_sim: float  # similarity to closest in CLIP
    t5_best_sim: float  # similarity to closest in T5

    @property
    def quadrant(self) -> str:
        """Classify based on dual-space confidence."""
        clip_ok = self.clip_confidence > self.clip_best_sim * 0.85 or self.clip_best_category == self.category
        t5_ok = self.t5_confidence > self.t5_best_sim * 0.85 or self.t5_best_category == self.category
        if clip_ok and t5_ok:
            return "rock_solid"
        if clip_ok and not t5_ok:
            return "visual_anchor"
        if not clip_ok and t5_ok:
            return "semantic_anchor"
        return "weak"


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: SET ANALYSIS (no models needed)
# ═══════════════════════════════════════════════════════════════════════════════


def set_analysis(pools: dict[str, dict[str, list[str]]]) -> dict[str, Any]:
    """Cross-pool overlap analysis. Pure set math, no embeddings."""
    pool_names = list(pools.keys())
    categories = sorted(set().union(*(set(p.keys()) for p in pools.values())))

    report: dict[str, Any] = {"pools": {}, "categories": {}, "pairwise": {}}

    # ── Global overview ──────────────────────────────────────────────
    print(f"\n{'='*75}")
    print("PHASE 1: CROSS-POOL SET ANALYSIS")
    print(f"{'='*75}")

    # Pool sizes
    for name, pool in pools.items():
        total = sum(len(v) for v in pool.values())
        report["pools"][name] = {
            "total": total,
            "categories": len(pool),
            "per_category": {c: len(pool.get(c, [])) for c in categories},
        }
        print(f"\n  {name}: {total} items across {len(pool)} categories")

    # ── Build flat word→categories maps ──────────────────────────────
    pool_word_cats: dict[str, dict[str, str]] = {}  # pool → {word: category}
    for name, pool in pools.items():
        wc: dict[str, str] = {}
        for cat, words in pool.items():
            for w in words:
                wc[w] = cat
        pool_word_cats[name] = wc

    # ── Pairwise overlap ─────────────────────────────────────────────
    print(f"\n{'─'*75}")
    print("PAIRWISE OVERLAP")
    print(f"{'─'*75}")

    for i, a in enumerate(pool_names):
        for b in pool_names[i + 1 :]:
            words_a = set(pool_word_cats[a].keys())
            words_b = set(pool_word_cats[b].keys())
            shared = words_a & words_b
            only_a = words_a - words_b
            only_b = words_b - words_a
            jaccard = len(shared) / len(words_a | words_b) if (words_a | words_b) else 0

            report["pairwise"][f"{a}_vs_{b}"] = {
                "shared": len(shared),
                "only_a": len(only_a),
                "only_b": len(only_b),
                "jaccard": round(jaccard, 4),
            }

            print(f"\n  {a} vs {b}:")
            print(f"    Shared: {len(shared)}  |  Only {a}: {len(only_a)}  |  Only {b}: {len(only_b)}")
            print(f"    Jaccard: {jaccard:.1%}")

            # Check category agreement for shared words
            if shared:
                same_cat = sum(1 for w in shared if pool_word_cats[a][w] == pool_word_cats[b][w])
                diff_cat = len(shared) - same_cat
                print(f"    Category agreement: {same_cat}/{len(shared)} ({same_cat / len(shared):.1%})")
                if diff_cat > 0:
                    print(f"    Category disagreements: {diff_cat}")
                    # Show first few disagreements
                    shown = 0
                    for w in sorted(shared):
                        if pool_word_cats[a][w] != pool_word_cats[b][w]:
                            print(f"      '{w}': {a}={pool_word_cats[a][w]}, {b}={pool_word_cats[b][w]}")
                            shown += 1
                            if shown >= 10:
                                remaining = diff_cat - shown
                                if remaining > 0:
                                    print(f"      ... and {remaining} more")
                                break

    # ── Consensus core (in all pools) ────────────────────────────────
    if len(pool_names) >= 2:
        print(f"\n{'─'*75}")
        print("CONSENSUS CORE (present in all pools)")
        print(f"{'─'*75}")

        all_word_sets = [set(pool_word_cats[n].keys()) for n in pool_names]
        consensus = all_word_sets[0]
        for ws in all_word_sets[1:]:
            consensus &= ws

        print(f"\n  {len(consensus)} words present in all {len(pool_names)} pools")

        # Break down by category (use first pool's categorization)
        consensus_by_cat: dict[str, list[str]] = {}
        for w in consensus:
            cat = pool_word_cats[pool_names[0]][w]
            consensus_by_cat.setdefault(cat, []).append(w)

        for cat in sorted(consensus_by_cat):
            words = sorted(consensus_by_cat[cat])
            print(f"    {cat}: {len(words)}")

        report["consensus_core"] = {
            "total": len(consensus),
            "by_category": {c: sorted(ws) for c, ws in sorted(consensus_by_cat.items())},
        }

    # ── Unique discoveries per pool ──────────────────────────────────
    print(f"\n{'─'*75}")
    print("UNIQUE TO EACH POOL")
    print(f"{'─'*75}")

    for name in pool_names:
        others = set()
        for other in pool_names:
            if other != name:
                others |= set(pool_word_cats[other].keys())
        unique = set(pool_word_cats[name].keys()) - others

        unique_by_cat: dict[str, list[str]] = {}
        for w in unique:
            cat = pool_word_cats[name][w]
            unique_by_cat.setdefault(cat, []).append(w)

        print(f"\n  {name}: {len(unique)} unique words")
        for cat in sorted(unique_by_cat):
            words = sorted(unique_by_cat[cat])
            preview = ", ".join(words[:5])
            more = f" (+{len(words) - 5} more)" if len(words) > 5 else ""
            print(f"    {cat} ({len(words)}): {preview}{more}")

        report.setdefault("unique", {})[name] = {
            "total": len(unique),
            "by_category": {c: sorted(ws) for c, ws in sorted(unique_by_cat.items())},
        }

    # ── Per-category cross-pool summary ──────────────────────────────
    print(f"\n{'─'*75}")
    print("PER-CATEGORY SUMMARY")
    print(f"{'─'*75}")

    header = f"  {'Category':<25}"
    for name in pool_names:
        header += f" {name:>12}"
    print(header)
    print(f"  {'-'*25}" + f" {'-'*12}" * len(pool_names))

    for cat in categories:
        row = f"  {cat:<25}"
        for name in pool_names:
            count = len(pools[name].get(cat, []))
            row += f" {count:>12}"
        print(row)

    # Totals
    row = f"  {'TOTAL':<25}"
    for name in pool_names:
        total = sum(len(v) for v in pools[name].values())
        row += f" {total:>12}"
    print(row)

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# EMBEDDING CACHE (compute once for all pools)
# ═══════════════════════════════════════════════════════════════════════════════


class AnalysisCache:
    """Persistent dual-space embedding cache. Compute once, reuse forever.

    Saves to a .npz file on disk so embeddings survive across runs.
    On subsequent loads, only computes embeddings for new words not already
    in the cache — avoids reloading 50GB+ of models for a few additions.
    """

    CACHE_PATH = DATA_DIR / "embedding_cache_openclip_t5xxl.npz"

    def __init__(
        self,
        all_words: list[str],
        embedder: DualSpaceEmbedder | None = None,
        cache_path: Path | None = None,
    ) -> None:
        self._cache_path = cache_path or self.CACHE_PATH
        self._clip: dict[str, np.ndarray] = {}
        self._t5: dict[str, np.ndarray | None] = {}
        self._has_t5 = False

        # Load existing cache from disk
        if self._cache_path.exists():
            self._load_from_disk()

        # Find words not yet cached
        unique = list(dict.fromkeys(all_words))
        missing = [w for w in unique if w not in self._clip]

        if missing and embedder is None:
            raise ValueError(
                f"{len(missing)} words need embedding but no embedder provided. "
                f"Pass an embedder or ensure all words are in the cache."
            )

        if missing and embedder is not None:
            logger.info(f"Cache hit: {len(unique) - len(missing)}/{len(unique)} words. "
                        f"Computing {len(missing)} new embeddings...")
            clip_new = embedder.clip_embedder.encode_batch(missing)
            for i, w in enumerate(missing):
                self._clip[w] = clip_new[i]

            if embedder.t5_embedder:
                t5_new = embedder.t5_embedder.encode_batch(missing)
                for i, w in enumerate(missing):
                    self._t5[w] = t5_new[i]
                self._has_t5 = True

            # Save updated cache to disk
            self._save_to_disk()
        elif not missing:
            logger.info(f"Full cache hit: all {len(unique)} words loaded from disk")

    def _load_from_disk(self) -> None:
        """Load cached embeddings from .npz file."""
        data = np.load(self._cache_path, allow_pickle=True)
        meta = {}
        if "_meta_json" in data:
            import json
            meta = json.loads(str(data["_meta_json"]))

        for key in data.files:
            if key.startswith("_"):
                continue
            if key.startswith("clip/"):
                word = key[5:]  # strip "clip/"
                self._clip[word] = data[key]
            elif key.startswith("t5/"):
                word = key[3:]  # strip "t5/"
                self._t5[word] = data[key]
                self._has_t5 = True

        logger.info(f"Loaded cache from {self._cache_path}: {len(self._clip)} words"
                     + (f" (clip={meta.get('clip_dim', '?')}d, t5={meta.get('t5_dim', '?')}d)" if meta else ""))

    def _save_to_disk(self) -> None:
        """Persist all cached embeddings to .npz file."""
        import json
        save_dict: dict[str, np.ndarray] = {}

        for word, emb in self._clip.items():
            save_dict[f"clip/{word}"] = emb
        for word, emb in self._t5.items():
            if emb is not None:
                save_dict[f"t5/{word}"] = emb

        # Metadata
        sample_clip = next(iter(self._clip.values()))
        clip_dim = sample_clip.shape[0]
        t5_dim = 0
        if self._has_t5:
            sample_t5 = next(v for v in self._t5.values() if v is not None)
            t5_dim = sample_t5.shape[0]

        meta = {
            "clip_model": "openclip-ViT-bigG-14",
            "t5_model": "google/t5-v1_1-xxl",
            "clip_dim": int(clip_dim),
            "t5_dim": int(t5_dim),
            "total_words": len(self._clip),
            "saved_at": datetime.now().isoformat(),
        }
        save_dict["_meta_json"] = np.array(json.dumps(meta))

        np.savez_compressed(self._cache_path, **save_dict)
        size_mb = self._cache_path.stat().st_size / (1024 * 1024)
        logger.info(f"Saved cache: {len(self._clip)} words → {self._cache_path} ({size_mb:.1f} MB)")

    def get_clip(self, words: list[str]) -> np.ndarray:
        try:
            return np.array([self._clip[w] for w in words])
        except KeyError as e:
            raise KeyError(f"Word not in cache: {e}. Rebuild cache with this word included.") from e

    def get_t5(self, words: list[str]) -> np.ndarray | None:
        if not self._has_t5:
            return None
        try:
            return np.array([self._t5[w] for w in words])
        except KeyError as e:
            raise KeyError(f"Word not in T5 cache: {e}. Rebuild cache with this word included.") from e

    def get_dual(self, words: list[str]) -> DualEmbeddings:
        clip = self.get_clip(words)
        t5 = self.get_t5(words)
        return DualEmbeddings(words=words, clip=clip, t5=t5)

    @property
    def has_t5(self) -> bool:
        return self._has_t5

    @property
    def cached_words(self) -> set[str]:
        return set(self._clip.keys())


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: DUAL-SPACE CONFIDENCE SCORING
# ═══════════════════════════════════════════════════════════════════════════════


def compute_centroids(
    pool: dict[str, list[str]],
    cache: AnalysisCache,
) -> dict[str, tuple[np.ndarray, np.ndarray | None]]:
    """Compute L2-normalized centroids per category in both spaces."""
    centroids: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
    for cat, words in pool.items():
        if not words:
            continue
        clip_emb = cache.get_clip(words)
        clip_c = clip_emb.mean(axis=0)
        clip_c /= np.linalg.norm(clip_c)

        t5_c = None
        t5_emb = cache.get_t5(words)
        if t5_emb is not None:
            t5_c = t5_emb.mean(axis=0)
            t5_c /= np.linalg.norm(t5_c)

        centroids[cat] = (clip_c, t5_c)
    return centroids


def confidence_scoring(
    pools: dict[str, dict[str, list[str]]],
    cache: AnalysisCache,
) -> dict[str, Any]:
    """Per-word confidence in both spaces for each pool."""
    report: dict[str, Any] = {}

    for pool_name, pool in pools.items():
        print(f"\n{'─'*75}")
        print(f"CONFIDENCE SCORING: {pool_name}")
        print(f"{'─'*75}")

        centroids = compute_centroids(pool, cache)
        cat_names = list(centroids.keys())

        pool_results: dict[str, Any] = {"categories": {}, "quadrants": {}, "flagged": []}
        all_confidences: list[WordConfidence] = []

        for cat, words in sorted(pool.items()):
            if not words or cat not in centroids:
                continue

            clip_c, t5_c = centroids[cat]
            clip_emb = cache.get_clip(words)

            # Cosine to own centroid
            clip_own = clip_emb @ clip_c

            # Cosine to all centroids in CLIP
            clip_all_sims = {}
            for other_cat, (other_clip_c, _) in centroids.items():
                clip_all_sims[other_cat] = clip_emb @ other_clip_c

            t5_own = np.zeros(len(words))
            t5_all_sims: dict[str, np.ndarray] = {}
            if t5_c is not None:
                t5_emb = cache.get_t5(words)
                if t5_emb is not None:
                    t5_own = t5_emb @ t5_c
                    for other_cat, (_, other_t5_c) in centroids.items():
                        if other_t5_c is not None:
                            t5_all_sims[other_cat] = t5_emb @ other_t5_c

            cat_confidences: list[WordConfidence] = []
            for i, word in enumerate(words):
                # Find best category in each space
                clip_best_cat = max(clip_all_sims, key=lambda c: clip_all_sims[c][i])
                clip_best_sim = float(clip_all_sims[clip_best_cat][i])

                t5_best_cat = cat
                t5_best_sim = float(t5_own[i])
                if t5_all_sims:
                    t5_best_cat = max(t5_all_sims, key=lambda c: t5_all_sims[c][i])
                    t5_best_sim = float(t5_all_sims[t5_best_cat][i])

                wc = WordConfidence(
                    word=word,
                    category=cat,
                    clip_confidence=float(clip_own[i]),
                    t5_confidence=float(t5_own[i]),
                    clip_best_category=clip_best_cat,
                    t5_best_category=t5_best_cat,
                    clip_best_sim=clip_best_sim,
                    t5_best_sim=t5_best_sim,
                )
                cat_confidences.append(wc)
                all_confidences.append(wc)

            clip_vals = [wc.clip_confidence for wc in cat_confidences]
            t5_vals = [wc.t5_confidence for wc in cat_confidences]

            pool_results["categories"][cat] = {
                "count": len(words),
                "clip_mean": round(float(np.mean(clip_vals)), 4),
                "clip_std": round(float(np.std(clip_vals)), 4),
                "clip_min": round(float(np.min(clip_vals)), 4),
                "t5_mean": round(float(np.mean(t5_vals)), 4),
                "t5_std": round(float(np.std(t5_vals)), 4),
                "t5_min": round(float(np.min(t5_vals)), 4),
            }

            # Flag words closer to wrong category
            misclassed = [wc for wc in cat_confidences
                          if wc.clip_best_category != cat or wc.t5_best_category != cat]
            for wc in misclassed:
                pool_results["flagged"].append({
                    "word": wc.word,
                    "assigned": wc.category,
                    "clip_says": wc.clip_best_category,
                    "t5_says": wc.t5_best_category,
                    "clip_conf": round(wc.clip_confidence, 4),
                    "t5_conf": round(wc.t5_confidence, 4),
                    "quadrant": wc.quadrant,
                })

        # Quadrant distribution
        quadrant_counts = {"rock_solid": 0, "visual_anchor": 0, "semantic_anchor": 0, "weak": 0}
        for wc in all_confidences:
            quadrant_counts[wc.quadrant] += 1

        total = len(all_confidences)
        pool_results["quadrants"] = {
            q: {"count": c, "pct": round(c / total * 100, 1) if total else 0}
            for q, c in quadrant_counts.items()
        }

        # Print summary
        print(f"\n  {'Category':<25} {'CLIP mean':>10} {'T5 mean':>10} {'CLIP min':>10} {'T5 min':>10}")
        print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        for cat in sorted(pool_results["categories"]):
            cs = pool_results["categories"][cat]
            print(f"  {cat:<25} {cs['clip_mean']:>10.4f} {cs['t5_mean']:>10.4f}"
                  f" {cs['clip_min']:>10.4f} {cs['t5_min']:>10.4f}")

        print(f"\n  Quadrant distribution:")
        for q, info in pool_results["quadrants"].items():
            bar = "#" * int(info["pct"] / 2)
            print(f"    {q:<18} {info['count']:>4} ({info['pct']:>5.1f}%) {bar}")

        flagged = pool_results["flagged"]
        if flagged:
            # Split into both-agree-wrong vs single-space-wrong
            both_wrong = [f for f in flagged if f["clip_says"] != f["assigned"] and f["t5_says"] != f["assigned"]]
            clip_wrong = [f for f in flagged if f["clip_says"] != f["assigned"] and f["t5_says"] == f["assigned"]]
            t5_wrong = [f for f in flagged if f["t5_says"] != f["assigned"] and f["clip_says"] == f["assigned"]]

            print(f"\n  Categorization flags:")
            print(f"    Both spaces disagree with assignment: {len(both_wrong)}")
            print(f"    CLIP only disagrees: {len(clip_wrong)}")
            print(f"    T5 only disagrees: {len(t5_wrong)}")

            if both_wrong:
                print(f"\n  Both spaces say wrong category (strongest candidates for re-categorization):")
                # Sort by how far off they are
                both_wrong.sort(key=lambda f: f["clip_conf"] + f["t5_conf"])
                for f in both_wrong[:15]:
                    print(f"    '{f['word']}': assigned={f['assigned']}, "
                          f"clip→{f['clip_says']} ({f['clip_conf']:.3f}), "
                          f"t5→{f['t5_says']} ({f['t5_conf']:.3f})")
                if len(both_wrong) > 15:
                    print(f"    ... and {len(both_wrong) - 15} more")

        report[pool_name] = pool_results

    # ── Cross-pool confidence comparison ─────────────────────────────
    if len(pools) >= 2:
        print(f"\n{'─'*75}")
        print("CROSS-POOL CONFIDENCE COMPARISON")
        print(f"{'─'*75}")

        names = list(pools.keys())
        categories = sorted(set().union(*(set(p.keys()) for p in pools.values())))

        print(f"\n  Mean CLIP confidence by category:")
        header = f"  {'Category':<25}"
        for n in names:
            header += f" {n:>12}"
        print(header)
        print(f"  {'-'*25}" + f" {'-'*12}" * len(names))

        for cat in categories:
            row = f"  {cat:<25}"
            for n in names:
                cs = report.get(n, {}).get("categories", {}).get(cat, {})
                val = cs.get("clip_mean", 0)
                row += f" {val:>12.4f}" if val else f" {'—':>12}"
            print(row)

        print(f"\n  Mean T5 confidence by category:")
        print(header)
        print(f"  {'-'*25}" + f" {'-'*12}" * len(names))

        for cat in categories:
            row = f"  {cat:<25}"
            for n in names:
                cs = report.get(n, {}).get("categories", {}).get(cat, {})
                val = cs.get("t5_mean", 0)
                row += f" {val:>12.4f}" if val else f" {'—':>12}"
            print(row)

        print(f"\n  Quadrant comparison:")
        header = f"  {'Quadrant':<18}"
        for n in names:
            header += f" {n:>12}"
        print(header)
        print(f"  {'-'*18}" + f" {'-'*12}" * len(names))

        for q in ["rock_solid", "visual_anchor", "semantic_anchor", "weak"]:
            row = f"  {q:<18}"
            for n in names:
                info = report.get(n, {}).get("quadrants", {}).get(q, {})
                pct = info.get("pct", 0)
                row += f" {pct:>10.1f}%"
            print(row)

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: INTER-CATEGORY GAP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════


def inter_category_gaps(
    pool: dict[str, list[str]],
    cache: AnalysisCache,
    pool_name: str = "gated",
) -> dict[str, Any]:
    """Centroid distance analysis in both spaces."""
    print(f"\n{'='*75}")
    print(f"PHASE 3: INTER-CATEGORY GAP ANALYSIS ({pool_name})")
    print(f"{'='*75}")

    centroids = compute_centroids(pool, cache)
    cat_names = sorted(centroids.keys())
    n = len(cat_names)

    # Build distance matrices
    clip_matrix = np.zeros((n, n))
    t5_matrix = np.zeros((n, n))
    has_t5 = cache.has_t5

    for i, a in enumerate(cat_names):
        for j, b in enumerate(cat_names):
            clip_a, t5_a = centroids[a]
            clip_b, t5_b = centroids[b]
            clip_matrix[i, j] = float(np.dot(clip_a, clip_b))
            if has_t5 and t5_a is not None and t5_b is not None:
                t5_matrix[i, j] = float(np.dot(t5_a, t5_b))

    # ── Print CLIP centroid matrix ───────────────────────────────────
    short = [c[:7] for c in cat_names]

    print(f"\n  OpenCLIP centroid similarities:")
    print(f"  {'':>16}", end="")
    for s in short:
        print(f" {s:>8}", end="")
    print()

    for i, name in enumerate(cat_names):
        print(f"  {short[i]:>16}", end="")
        for j in range(n):
            if i == j:
                print(f"    —   ", end="")
            else:
                print(f" {clip_matrix[i, j]:>7.3f}", end="")
        print()

    # ── Print T5 centroid matrix ─────────────────────────────────────
    if has_t5:
        print(f"\n  T5-XXL centroid similarities:")
        print(f"  {'':>16}", end="")
        for s in short:
            print(f" {s:>8}", end="")
        print()

        for i, name in enumerate(cat_names):
            print(f"  {short[i]:>16}", end="")
            for j in range(n):
                if i == j:
                    print(f"    —   ", end="")
                else:
                    print(f" {t5_matrix[i, j]:>7.3f}", end="")
            print()

    # ── Closest and farthest pairs ───────────────────────────────────
    print(f"\n{'─'*75}")
    print("CLOSEST CATEGORY PAIRS (merge candidates)")
    print(f"{'─'*75}")

    pairs: list[tuple[str, str, float, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            clip_sim = clip_matrix[i, j]
            t5_sim = t5_matrix[i, j] if has_t5 else 0
            pairs.append((cat_names[i], cat_names[j], clip_sim, t5_sim))

    # Sort by average proximity
    if has_t5:
        pairs.sort(key=lambda p: (p[2] + p[3]) / 2, reverse=True)
    else:
        pairs.sort(key=lambda p: p[2], reverse=True)

    print(f"\n  {'Pair':<50} {'CLIP':>7} {'T5':>7} {'Avg':>7}")
    print(f"  {'-'*50} {'-'*7} {'-'*7} {'-'*7}")
    for a, b, cs, ts in pairs[:10]:
        avg = (cs + ts) / 2 if has_t5 else cs
        print(f"  {a} ↔ {b:<25} {cs:>7.3f} {ts:>7.3f} {avg:>7.3f}")

    print(f"\n{'─'*75}")
    print("MOST DISTANT CATEGORY PAIRS (strongest axes)")
    print(f"{'─'*75}")

    if has_t5:
        pairs.sort(key=lambda p: (p[2] + p[3]) / 2)
    else:
        pairs.sort(key=lambda p: p[2])

    print(f"\n  {'Pair':<50} {'CLIP':>7} {'T5':>7} {'Avg':>7}")
    print(f"  {'-'*50} {'-'*7} {'-'*7} {'-'*7}")
    for a, b, cs, ts in pairs[:10]:
        avg = (cs + ts) / 2 if has_t5 else cs
        print(f"  {a} ↔ {b:<25} {cs:>7.3f} {ts:>7.3f} {avg:>7.3f}")

    # ── Space disagreements ──────────────────────────────────────────
    if has_t5:
        print(f"\n{'─'*75}")
        print("SPACE DISAGREEMENTS (CLIP vs T5 see categories differently)")
        print(f"{'─'*75}")

        disagreements: list[tuple[str, str, float, float, float]] = []
        for a, b, cs, ts in pairs:
            delta = abs(cs - ts)
            disagreements.append((a, b, cs, ts, delta))

        disagreements.sort(key=lambda x: x[4], reverse=True)

        print(f"\n  {'Pair':<50} {'CLIP':>7} {'T5':>7} {'|Δ|':>7}")
        print(f"  {'-'*50} {'-'*7} {'-'*7} {'-'*7}")
        for a, b, cs, ts, delta in disagreements[:10]:
            direction = "CLIP closer" if cs > ts else "T5 closer"
            print(f"  {a} ↔ {b:<25} {cs:>7.3f} {ts:>7.3f} {delta:>7.3f}  ({direction})")

    # ── Gap analysis: where might a 13th axis live? ──────────────────
    print(f"\n{'─'*75}")
    print("GAP ANALYSIS")
    print(f"{'─'*75}")

    # Categories with highest average distance from all others = most isolated
    isolation: list[tuple[str, float, float]] = []
    for i, cat in enumerate(cat_names):
        clip_dists = [clip_matrix[i, j] for j in range(n) if j != i]
        t5_dists = [t5_matrix[i, j] for j in range(n) if j != i] if has_t5 else [0]
        isolation.append((cat, float(np.mean(clip_dists)), float(np.mean(t5_dists))))

    isolation.sort(key=lambda x: (x[1] + x[2]) / 2 if has_t5 else x[1])

    print(f"\n  Category isolation (lower avg similarity = more distinct):")
    print(f"  {'Category':<25} {'CLIP avg':>10} {'T5 avg':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10}")
    for cat, clip_avg, t5_avg in isolation:
        print(f"  {cat:<25} {clip_avg:>10.4f} {t5_avg:>10.4f}")

    # Build report
    report: dict[str, Any] = {
        "clip_centroids": {
            cat_names[i]: {cat_names[j]: round(clip_matrix[i, j], 4) for j in range(n)}
            for i in range(n)
        },
        "closest_pairs": [
            {"a": a, "b": b, "clip": round(cs, 4), "t5": round(ts, 4)}
            for a, b, cs, ts in sorted(pairs, key=lambda p: (p[2] + p[3]) / 2 if has_t5 else p[2], reverse=True)[:10]
        ],
        "most_distant": [
            {"a": a, "b": b, "clip": round(cs, 4), "t5": round(ts, 4)}
            for a, b, cs, ts in sorted(pairs, key=lambda p: (p[2] + p[3]) / 2 if has_t5 else p[2])[:10]
        ],
        "isolation": [{"category": cat, "clip_avg": round(c, 4), "t5_avg": round(t, 4)} for cat, c, t in isolation],
    }

    if has_t5:
        report["t5_centroids"] = {
            cat_names[i]: {cat_names[j]: round(t5_matrix[i, j], 4) for j in range(n)}
            for i in range(n)
        }
        report["disagreements"] = [
            {"a": a, "b": b, "clip": round(cs, 4), "t5": round(ts, 4), "delta": round(d, 4)}
            for a, b, cs, ts, d in disagreements[:15]
        ]

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-pool analysis and dual-space confidence scoring",
    )
    parser.add_argument(
        "--set-only", action="store_true",
        help="Phase 1 only — set analysis, no embedding models needed",
    )
    parser.add_argument(
        "--pools", nargs="+", choices=list(POOL_PATHS.keys()), default=["original", "curated_v2", "gated"],
        help="Which pools to analyze (default: all three)",
    )
    parser.add_argument(
        "--clip-model", default="openclip",
        help="CLIP model for embedding (default: openclip = ViT-bigG-14)",
    )
    parser.add_argument(
        "--t5-model", default="google/t5-v1_1-xxl",
        help="T5 model for embedding (default: google/t5-v1_1-xxl)",
    )
    parser.add_argument(
        "--gap-pool", default="gated",
        help="Which pool to use for inter-category gap analysis (default: gated)",
    )
    parser.add_argument(
        "--output", "-o", type=Path,
        help="Save full report to YAML",
    )
    parser.add_argument(
        "--cache", type=str,
        help=f"Path to embedding cache .npz (default: {AnalysisCache.CACHE_PATH})",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S",
    )

    # Load pools
    pools: dict[str, dict[str, list[str]]] = {}
    for name in args.pools:
        path = POOL_PATHS[name]
        if not path.exists():
            logger.error(f"Pool not found: {path}")
            sys.exit(1)
        pools[name] = load_components_yaml(path)

    total_words = sum(sum(len(v) for v in p.values()) for p in pools.values())
    print(f"{'='*75}")
    print("CROSS-POOL ANALYSIS & CONFIDENCE SCORING")
    print(f"{'='*75}")
    print(f"  Pools: {', '.join(args.pools)}")
    print(f"  Total words across pools: {total_words}")

    full_report: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "pools_analyzed": args.pools,
    }

    # Phase 1: Set analysis (always runs)
    set_report = set_analysis(pools)
    full_report["set_analysis"] = set_report

    if args.set_only:
        print(f"\n{'='*75}")
        print("Done (set analysis only)")
        if args.output:
            with open(args.output, "w") as f:
                yaml.dump(full_report, f, default_flow_style=False, allow_unicode=True, width=120)
            print(f"Report saved: {args.output}")
        return

    # Collect all unique words across all pools
    all_words: list[str] = []
    for pool in pools.values():
        for words in pool.values():
            all_words.extend(words)
    unique_words = list(dict.fromkeys(all_words))

    print(f"\n{'='*75}")
    print("PHASE 2: DUAL-SPACE CONFIDENCE SCORING")
    print(f"{'='*75}")
    print(f"  CLIP model: {args.clip_model}")
    print(f"  T5 model: {args.t5_model}")
    print(f"  Unique words to embed: {len(unique_words)}")

    # Try loading from disk cache first — only load models if we have missing words
    cache_path = Path(args.cache) if args.cache else AnalysisCache.CACHE_PATH
    embedder: DualSpaceEmbedder | None = None

    # Check what's already cached
    cached_words: set[str] = set()
    if cache_path.exists():
        try:
            existing = np.load(cache_path, allow_pickle=True)
            cached_words = {k[5:] for k in existing.files if k.startswith("clip/")}
            existing.close()
        except Exception:
            pass

    missing = [w for w in unique_words if w not in cached_words]
    if missing:
        print(f"  Cache has {len(cached_words)} words, need {len(missing)} new embeddings")
        print(f"  Loading models...")
        embedder = DualSpaceEmbedder(clip_model=args.clip_model, t5_model=args.t5_model)
    else:
        print(f"  Full cache hit — {len(cached_words)} words, no model load needed")

    cache = AnalysisCache(unique_words, embedder=embedder, cache_path=cache_path)

    # Free model memory
    if embedder is not None:
        del embedder

    # Phase 2: Confidence scoring
    confidence_report = confidence_scoring(pools, cache)
    full_report["confidence"] = confidence_report

    # Phase 3: Inter-category gaps
    gap_pool_name = args.gap_pool
    if gap_pool_name in pools:
        gap_report = inter_category_gaps(pools[gap_pool_name], cache, gap_pool_name)
        full_report["inter_category_gaps"] = gap_report
    else:
        logger.warning(f"Gap pool '{gap_pool_name}' not in analyzed pools, skipping Phase 3")

    # Save report
    if args.output:
        with open(args.output, "w") as f:
            yaml.dump(full_report, f, default_flow_style=False, allow_unicode=True, width=120)
        print(f"\nReport saved: {args.output}")

    print(f"\n{'='*75}")
    print("Done!")


if __name__ == "__main__":
    main()
