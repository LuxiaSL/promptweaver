#!/usr/bin/env python3
"""
Embedding Space Visualization
═══════════════════════════════════════════════════════════════════════════════

Generates PCA/UMAP plots of the embedding space for visual inspection:
- Category cluster scatter (CLIP, T5, joint)
- Inter-category centroid distance heatmap
- Per-category spread comparison

Saves PNG files — no display needed (works on headless nodes).

Usage:
    uv run python -m promptweaver.tools.visualize \
        -i data/selected_final.yaml \
        --out-dir data/plots/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Headless — no display needed
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from sklearn.decomposition import PCA

from .embeddings import (
    DualSpaceEmbedder,
    DualEmbeddings,
    load_components_yaml,
)

logger = logging.getLogger(__name__)

# Consistent colors per category
CATEGORY_COLORS: dict[str, str] = {
    "subject_form": "#e6194b",
    "material_substance": "#f58231",
    "texture_density": "#ffe119",
    "light_behavior": "#bfef45",
    "color_logic": "#3cb44b",
    "atmosphere_field": "#42d4f4",
    "phenomenon_pattern": "#4363d8",
    "spatial_logic": "#911eb4",
    "scale_perspective": "#f032e6",
    "temporal_state": "#a9a9a9",
    "setting_location": "#9a6324",
    "medium_render": "#800000",
}


def plot_scatter(
    embeddings_2d: np.ndarray,
    words: list[str],
    categories: list[str],
    title: str,
    path: Path,
    show_labels: bool = False,
) -> None:
    """2D scatter plot colored by category."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    unique_cats = sorted(set(categories))
    for cat in unique_cats:
        mask = [c == cat for c in categories]
        pts = embeddings_2d[mask]
        color = CATEGORY_COLORS.get(cat, "#333333")
        ax.scatter(
            pts[:, 0], pts[:, 1],
            c=[color], label=cat, alpha=0.6, s=30, edgecolors="white", linewidths=0.3,
        )

        if show_labels:
            cat_words = [w for w, c in zip(words, categories) if c == cat]
            for j, (x, y) in enumerate(pts):
                ax.annotate(
                    cat_words[j], (x, y),
                    fontsize=4, alpha=0.5, ha="center", va="bottom",
                )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(
        bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8,
        markerscale=1.5, framealpha=0.9,
    )
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {path}")


def plot_centroid_heatmap(
    centroids: dict[str, np.ndarray],
    title: str,
    path: Path,
) -> None:
    """Pairwise centroid similarity heatmap."""
    cats = sorted(centroids.keys())
    n = len(cats)
    sim = np.zeros((n, n))

    for i, a in enumerate(cats):
        for j, b in enumerate(cats):
            sim[i, j] = float(np.dot(centroids[a], centroids[b]))

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    im = ax.imshow(sim, cmap="RdYlBu_r", vmin=0, vmax=1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    short_names = [c.replace("_", "\n") for c in cats]
    ax.set_xticklabels(short_names, fontsize=7, rotation=45, ha="right")
    ax.set_yticklabels(short_names, fontsize=7)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            color = "white" if sim[i, j] > 0.7 else "black"
            ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color=color)

    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Cosine Similarity")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {path}")


def plot_spread_comparison(
    cat_stats: dict[str, dict[str, float]],
    path: Path,
) -> None:
    """Bar chart comparing CLIP vs T5 variance per category."""
    cats = sorted(cat_stats.keys())
    clip_vars = [cat_stats[c]["clip_std"] for c in cats]
    t5_vars = [cat_stats[c]["t5_std"] for c in cats]

    x = np.arange(len(cats))
    width = 0.35

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.bar(x - width / 2, clip_vars, width, label="CLIP std", color="#4363d8", alpha=0.8)
    ax.bar(x + width / 2, t5_vars, width, label="T5 std", color="#e6194b", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in cats], fontsize=8)
    ax.set_ylabel("Pairwise Similarity Std Dev")
    ax.set_title("CLIP vs T5 Discriminative Power (higher std = more spread)", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {path}")


def try_umap(data: np.ndarray, n_neighbors: int = 15) -> np.ndarray | None:
    """Try UMAP reduction, fall back to None if not available or fails."""
    try:
        import umap
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
        return reducer.fit_transform(data)
    except Exception as e:
        logger.warning(f"  UMAP failed ({e}), skipping")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize embedding space")
    parser.add_argument("--input", "-i", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory for plots (default: data/plots/)")
    parser.add_argument("--clip-only", action="store_true")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--t5-model", default="google/flan-t5-large")
    parser.add_argument("--labels", action="store_true",
                        help="Show word labels on scatter plots (crowded but useful)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Joint space alpha for combined plot")
    parser.add_argument("--alpha-per-category", type=Path)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S",
    )

    if not args.input.exists():
        logger.error(f"Input not found: {args.input}")
        sys.exit(1)

    out_dir = args.out_dir or (Path(args.input).parent / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    t5_model = None if args.clip_only else args.t5_model

    per_cat_alpha: dict[str, float] = {}
    if args.alpha_per_category and args.alpha_per_category.exists():
        import json
        per_cat_alpha = json.loads(args.alpha_per_category.read_text())

    # Load and embed
    categories = load_components_yaml(args.input)
    total = sum(len(v) for v in categories.values())
    print(f"Visualizing: {len(categories)} categories, {total} words")

    embedder = DualSpaceEmbedder(clip_model=args.clip_model, t5_model=t5_model)

    # Collect all words with category labels
    all_words: list[str] = []
    all_cats: list[str] = []
    for cat, words in sorted(categories.items()):
        all_words.extend(words)
        all_cats.extend([cat] * len(words))

    # Embed
    clip_emb = embedder.clip_embedder.encode_batch(all_words)
    t5_emb = embedder.t5_embedder.encode_batch(all_words) if embedder.t5_embedder else None

    # ── 3D INTERACTIVE (plotly) ─────────────────────────────────────────
    print("\nGenerating 3D interactive plots...")
    try:
        import plotly.graph_objects as go

        def make_3d_scatter(emb_data: np.ndarray, title: str, path: Path, method: str = "pca") -> None:
            if method == "umap":
                try:
                    import umap
                    reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42)
                    pts = reducer.fit_transform(emb_data)
                except Exception:
                    logger.warning("  UMAP 3D failed, falling back to PCA")
                    pca = PCA(n_components=3, random_state=42)
                    pts = pca.fit_transform(emb_data)
            else:
                pca = PCA(n_components=3, random_state=42)
                pts = pca.fit_transform(emb_data)

            fig = go.Figure()
            for cat in sorted(set(all_cats)):
                mask = [c == cat for c in all_cats]
                cat_pts = pts[np.array(mask)]
                cat_words = [w for w, c in zip(all_words, all_cats) if c == cat]
                fig.add_trace(go.Scatter3d(
                    x=cat_pts[:, 0], y=cat_pts[:, 1], z=cat_pts[:, 2],
                    mode="markers",
                    name=cat,
                    text=cat_words,
                    hovertemplate="%{text}<br>%{fullData.name}<extra></extra>",
                    marker=dict(
                        size=3,
                        color=CATEGORY_COLORS.get(cat, "#333"),
                        opacity=0.7,
                    ),
                ))
            fig.update_layout(
                title=title,
                scene=dict(xaxis_title="C1", yaxis_title="C2", zaxis_title="C3"),
                width=1200, height=900,
                legend=dict(itemsizing="constant"),
            )
            fig.write_html(str(path))
            logger.info(f"  Saved: {path}")

        # CLIP 3D PCA
        make_3d_scatter(clip_emb, "CLIP Embedding Space (3D PCA)", out_dir / "3d_clip_pca.html", "pca")

        if t5_emb is not None:
            make_3d_scatter(t5_emb, "T5 Embedding Space (3D PCA)", out_dir / "3d_t5_pca.html", "pca")

        # UMAP 3D
        make_3d_scatter(clip_emb, "CLIP Embedding Space (3D UMAP)", out_dir / "3d_clip_umap.html", "umap")

        if t5_emb is not None:
            make_3d_scatter(t5_emb, "T5 Embedding Space (3D UMAP)", out_dir / "3d_t5_umap.html", "umap")

    except ImportError:
        print("  plotly not installed, skipping 3D interactive plots")

    # ── PCA SCATTER: CLIP SPACE ─────────────────────────────────────────
    print("\nGenerating CLIP space PCA...")
    pca_clip = PCA(n_components=2, random_state=42)
    clip_2d = pca_clip.fit_transform(clip_emb)
    var_clip = pca_clip.explained_variance_ratio_
    plot_scatter(
        clip_2d, all_words, all_cats,
        f"CLIP Embedding Space (PCA, var={var_clip[0]:.1%}+{var_clip[1]:.1%})",
        out_dir / "scatter_clip_pca.png",
        show_labels=args.labels,
    )

    # ── PCA SCATTER: T5 SPACE ───────────────────────────────────────────
    if t5_emb is not None:
        print("Generating T5 space PCA...")
        pca_t5 = PCA(n_components=2, random_state=42)
        t5_2d = pca_t5.fit_transform(t5_emb)
        var_t5 = pca_t5.explained_variance_ratio_
        plot_scatter(
            t5_2d, all_words, all_cats,
            f"T5 Embedding Space (PCA, var={var_t5[0]:.1%}+{var_t5[1]:.1%})",
            out_dir / "scatter_t5_pca.png",
            show_labels=args.labels,
        )

    # ── UMAP SCATTER ────────────────────────────────────────────────────
    print("Generating UMAP projections...")
    clip_umap = try_umap(clip_emb)
    if clip_umap is not None:
        plot_scatter(
            clip_umap, all_words, all_cats,
            "CLIP Embedding Space (UMAP)",
            out_dir / "scatter_clip_umap.png",
            show_labels=args.labels,
        )

    if t5_emb is not None:
        t5_umap = try_umap(t5_emb)
        if t5_umap is not None:
            plot_scatter(
                t5_umap, all_words, all_cats,
                "T5 Embedding Space (UMAP)",
                out_dir / "scatter_t5_umap.png",
                show_labels=args.labels,
            )

    # ── CENTROID HEATMAPS ───────────────────────────────────────────────
    print("Generating centroid heatmaps...")

    # CLIP centroids
    clip_centroids: dict[str, np.ndarray] = {}
    offset = 0
    for cat, words in sorted(categories.items()):
        n = len(words)
        c = clip_emb[offset:offset + n].mean(axis=0)
        clip_centroids[cat] = c / np.linalg.norm(c)
        offset += n

    plot_centroid_heatmap(
        clip_centroids,
        "Inter-Category Centroid Similarity (CLIP)",
        out_dir / "heatmap_clip_centroids.png",
    )

    if t5_emb is not None:
        t5_centroids: dict[str, np.ndarray] = {}
        offset = 0
        for cat, words in sorted(categories.items()):
            n = len(words)
            c = t5_emb[offset:offset + n].mean(axis=0)
            t5_centroids[cat] = c / np.linalg.norm(c)
            offset += n

        plot_centroid_heatmap(
            t5_centroids,
            "Inter-Category Centroid Similarity (T5)",
            out_dir / "heatmap_t5_centroids.png",
        )

    # ── SPREAD COMPARISON ───────────────────────────────────────────────
    if t5_emb is not None:
        print("Generating spread comparison...")
        cat_stats: dict[str, dict[str, float]] = {}
        offset = 0
        for cat, words in sorted(categories.items()):
            n = len(words)
            c_emb = clip_emb[offset:offset + n]
            t_emb = t5_emb[offset:offset + n]

            mask = ~np.eye(n, dtype=bool)
            clip_sim = (c_emb @ c_emb.T)[mask]
            t5_sim = (t_emb @ t_emb.T)[mask]

            cat_stats[cat] = {
                "clip_std": float(clip_sim.std()),
                "t5_std": float(t5_sim.std()),
            }
            offset += n

        plot_spread_comparison(cat_stats, out_dir / "spread_clip_vs_t5.png")

    print(f"\nAll plots saved to: {out_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
