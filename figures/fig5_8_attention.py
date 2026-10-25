#!/usr/bin/env python3
"""Figures 5-8: final-layer attention maps for L-GATr, L-GATr-slim and LLoCa-T, head-averaged and per head."""

from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
INTERP = ROOT / "results" / "interp"
OUT_DIR = ROOT / "figures" / "appendix"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("lgatr",             "avg_attn_lgatr_all.npy",             "L-GATr"),
    ("lgatr_slim",        "avg_attn_slim_all.npy",              "L-GATr-slim"),
    ("lloca_transformer", "avg_attn_lloca_transformer_all.npy", "LLoCa-T"),
]
SEEDS = ["seed1001", "seed1002", "seed1003"]
N_HEADS = 8

# Load all attention arrays: dict[(model_key, seed)] -> (8, 50, 50)
attn_data = {}
for model_key, attn_file, _ in MODELS:
    for seed in SEEDS:
        attn_data[(model_key, seed)] = np.load(INTERP / model_key / seed / attn_file)


def _imshow_cell(ax, mat, show_cbar=True):
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0)
    if show_cbar:
        plt.colorbar(im, ax=ax, shrink=0.82, pad=0.03, format="%.3f")
    ax.set_xticks([])
    ax.set_yticks([])
    return im


# ── figA1a: overview 3×3 (head-averaged) ──────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(10, 9))
for row, seed in enumerate(SEEDS):
    for col, (model_key, _, model_label) in enumerate(MODELS):
        ax = axes[row, col]
        avg = attn_data[(model_key, seed)].mean(axis=0)  # (50, 50)
        _imshow_cell(ax, avg, show_cbar=True)
        if row == 0:
            ax.set_xlabel(model_label, fontsize=11, labelpad=6)
            ax.xaxis.set_label_position("top")
        if col == 0:
            ax.set_ylabel(seed, fontsize=9)

fig.tight_layout()
for stem in ("figA1a_attention_overview",):
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=150, bbox_inches="tight")
    print(f"Saved {OUT_DIR / stem}.pdf")
plt.close(fig)


# ── figA1b/c/d: per-model, 3 seeds × 8 heads ──────────────────────────────────
for model_key, _, model_label in MODELS:
    tag = model_key.replace("_transformer", "").replace("lgatr_slim", "slim")
    # Landscape: 3 rows (seeds) × 8 cols (heads)
    fig, axes = plt.subplots(3, N_HEADS, figsize=(14, 6.5),
                             gridspec_kw=dict(hspace=0.08, wspace=0.06),
                             constrained_layout=True)
    for row, seed in enumerate(SEEDS):
        attn = attn_data[(model_key, seed)]  # (8, 50, 50)
        for h in range(N_HEADS):
            ax = axes[row, h]
            im = ax.imshow(attn[h], aspect="equal", interpolation="nearest",
                           cmap="viridis", vmin=0)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_xlabel(f"Head {h}", fontsize=8, labelpad=4)
                ax.xaxis.set_label_position("top")
            if h == 0:
                ax.set_ylabel(seed, fontsize=8)
        # One colorbar per row, stealing space equally from all 8 heads
        fig.colorbar(im, ax=axes[row, :], shrink=0.85, pad=0.03, format="%.3f")

    fig.suptitle(model_label, fontsize=11)
    for stem in (f"figA1{'bcd'[['lgatr','lgatr_slim','lloca_transformer'].index(model_key)]}_attention_{tag}",):
        fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(OUT_DIR / f"{stem}.png", dpi=150, bbox_inches="tight")
        print(f"Saved {OUT_DIR / stem}.pdf")
    plt.close(fig)
