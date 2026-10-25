#!/usr/bin/env python3
"""Figure 3: scalar-channel probe trajectories for particle eta, tau32 and jet mass across all five models."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
PROBES = ROOT / "results" / "probes"
OUT_DIR = ROOT / "figures" / "main"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── model metadata ─────────────────────────────────────────────────────────
MODELS = {
    "lgatr":               {"label": "L-GATr",      "color": "#409107", "lw": 2.0, "ls": "-"},
    "lgatr_slim":          {"label": "L-GATr-slim",  "color": "#1d828b", "lw": 2.0, "ls": "--"},
    "lloca_transformer":   {"label": "LLoCa-T",      "color": "#8b261d", "lw": 2.0, "ls": "-."},
    "part":                {"label": "ParT",          "color": "#e8c469", "lw": 1.5, "ls": "--"},
    "vanilla_transformer": {"label": "Vanilla-T",    "color": "#e16c5b", "lw": 1.5, "ls": ":"},
}

TARGETS = ["part_eta", "jet_tau32", "jet_mass"]
YLABELS = {
    "part_eta":  r"$R^2$ (particle $\eta$)",
    "jet_tau32": r"$R^2$ (jet $\tau_{32}$)",
    "jet_mass":  r"$R^2$ (jet mass)",
}
YLIMS = {
    "part_eta":  (None, None),
    "jet_tau32": (None, None),
    "jet_mass":  (0.94, 1.005),
}


def load_combined_ci(model_key: str, target: str):
    """Load combined bootstrap CI (median, lo, hi); each shape (n_layers,)."""
    f = np.load(PROBES / model_key / "combined_bootstrap_ci.npz")
    median = f[f"{target}__median"]
    lo     = f[f"{target}__lo"]
    hi     = f[f"{target}__hi"]
    return median, lo, hi


# ── figure ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4))

for ax, target in zip(axes, TARGETS):
    for model_key, meta in MODELS.items():
        median, lo, hi = load_combined_ci(model_key, target)
        n_layers = len(median)
        x = np.arange(n_layers)

        ax.plot(x, median, color=meta["color"], lw=meta["lw"], ls=meta["ls"],
                label=meta["label"], zorder=3)
        ax.fill_between(x, lo, hi, color=meta["color"], alpha=0.15, zorder=2)

    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel(YLABELS[target], fontsize=11)
    ax.set_xticks(range(0, n_layers, 2))
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.3)
    ylim = YLIMS[target]
    if ylim[0] is not None or ylim[1] is not None:
        ax.set_ylim(ylim)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=9,
           framealpha=0.9, bbox_to_anchor=(0.5, 1.02))

fig.tight_layout(rect=[0, 0, 1, 0.96])

out_png = OUT_DIR / "fig2_hs_probe_summary.png"
out_pdf = OUT_DIR / "fig2_hs_probe_summary.pdf"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
print(f"Saved {out_pdf}")
