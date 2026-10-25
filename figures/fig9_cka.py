#!/usr/bin/env python3
"""Figure 9: linear CKA between scalar-channel representations at all layer pairs."""

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
    ("lgatr",             "cka_lgatr.npy",             "L-GATr"),
    ("lgatr_slim",        "cka_slim.npy",               "L-GATr-slim"),
    ("lloca_transformer", "cka_lloca_transformer.npy",  "LLoCa-T"),
]
SEEDS = ["seed1001", "seed1002", "seed1003"]
LAYER_LABELS = [f"L{i}" for i in range(12)]   # L0–L11

fig, axes = plt.subplots(3, 3, figsize=(10, 9),
                         gridspec_kw=dict(hspace=0.12, wspace=0.08))

tick_pos = list(range(12))
ims = []

for row, (model_key, cka_file, model_label) in enumerate(MODELS):
    for col, seed in enumerate(SEEDS):
        ax = axes[row, col]
        mat_full = np.load(INTERP / model_key / seed / cka_file)  # (13, 13)
        mat = mat_full[1:, 1:]   # skip input-projection row/col → 12×12 (L0–L11)

        im = ax.imshow(mat, aspect="equal", interpolation="nearest",
                       cmap="viridis", vmin=0, vmax=1, origin="lower")
        ims.append(im)

        # Y-axis ticks: left column only
        ax.set_yticks(tick_pos)
        if col == 0:
            ax.set_yticklabels(LAYER_LABELS, fontsize=6)
            ax.set_ylabel(model_label, fontsize=10, labelpad=6)
        else:
            ax.set_yticklabels([])

        # X-axis ticks: bottom row only, at bottom
        ax.set_xticks(tick_pos)
        if row == 2:
            ax.set_xticklabels(LAYER_LABELS, fontsize=6, rotation=90)
            ax.tick_params(bottom=True, labelbottom=True, top=False, labeltop=False)
        else:
            ax.set_xticklabels([])
            ax.tick_params(bottom=False, labelbottom=False, top=False, labeltop=False)

        # Seed label: top row only, at top
        if row == 0:
            ax.set_xlabel(seed, fontsize=9, labelpad=5)
            ax.xaxis.set_label_position("top")

# Single shared colorbar to the right of all panels (no space stolen from any cell)
fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.90, 0.12, 0.02, 0.76])
fig.colorbar(ims[-1], cax=cbar_ax)
cbar_ax.tick_params(labelsize=8)

out_pdf = OUT_DIR / "figA2_cka_compare.pdf"
out_png = OUT_DIR / "figA2_cka_compare.png"
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"Saved {out_pdf}")
