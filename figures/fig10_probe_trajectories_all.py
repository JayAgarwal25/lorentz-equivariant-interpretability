#!/usr/bin/env python3
"""Figure 10: all remaining scalar-channel probe trajectories across all five models."""

from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT   = Path(__file__).parent.parent
PROBES = ROOT / "results" / "probes"
OUT_DIR = ROOT / "figures" / "appendix"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "lgatr":               {"label": "L-GATr",      "color": "#409107", "lw": 2.0, "ls": "-"},
    "lgatr_slim":          {"label": "L-GATr-slim",  "color": "#1d828b", "lw": 2.0, "ls": "--"},
    "lloca_transformer":   {"label": "LLoCa-T",      "color": "#8b261d", "lw": 2.0, "ls": "-."},
    "vanilla_transformer": {"label": "Vanilla-T",    "color": "#e16c5b", "lw": 1.5, "ls": ":"},
    "part":                {"label": "ParT",          "color": "#e8c469", "lw": 1.5, "ls": "--"},
}

TARGETS = [
    ("top_qcd",       r"AUC (top/QCD)"),
    ("jet_mass",      r"$R^2$ (jet mass)"),
    ("jet_mult",      r"$R^2$ (jet mult)"),
    ("jet_tau1",      r"$R^2$ (jet $\tau_1$)"),
    ("jet_tau2",      r"$R^2$ (jet $\tau_2$)"),
    ("jet_tau3",      r"$R^2$ (jet $\tau_3$)"),
    ("jet_tau21",     r"$R^2$ (jet $\tau_{21}$)"),
    ("part_pt",       r"$R^2$ (particle $p_T$)"),
    ("part_E",        r"$R^2$ (particle $E$)"),
    ("part_phi",      r"$R^2$ (particle $\phi$)"),
    ("part_quartile", r"Acc. (particle $p_T$-quartile)"),
    ("part_dR",       r"$R^2$ (particle $\Delta R$)"),
]


def load_combined_ci(model_key: str, target: str):
    f = np.load(PROBES / model_key / "combined_bootstrap_ci.npz")
    return f[f"{target}__median"], f[f"{target}__lo"], f[f"{target}__hi"]


fig, axes = plt.subplots(4, 3, figsize=(12, 13))

handles, labels = None, None

for idx, (target, ylabel) in enumerate(TARGETS):
    ax = axes[idx // 3, idx % 3]
    n_layers = None
    all_lo, all_hi = [], []

    for model_key, meta in MODELS.items():
        median, lo, hi = load_combined_ci(model_key, target)
        n_layers = len(median)
        x = np.arange(n_layers)
        # zoom based on converged region (last half of layers only)
        half = n_layers // 2
        all_lo.append(lo[half:].min())
        all_hi.append(hi[half:].max())

        line, = ax.plot(x, median, color=meta["color"], lw=meta["lw"], ls=meta["ls"],
                        label=meta["label"], zorder=3)
        ax.fill_between(x, lo, hi, color=meta["color"], alpha=0.15, zorder=2)

    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if n_layers is not None:
        ax.set_xticks(range(0, n_layers, 2))
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.3)

    # zoom y-axis to converged data range, cap top at 1.0
    ymin, ymax = min(all_lo), max(all_hi)
    margin = (ymax - ymin) * 0.20
    ax.set_ylim(ymin - margin, min(ymax + margin * 0.3, 1.0))

    # collect handles from the first panel
    if idx == 0:
        handles, labels = ax.get_legend_handles_labels()

# single shared legend below all panels
fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=10,
           framealpha=0.9, bbox_to_anchor=(0.5, 1.02))

fig.tight_layout(rect=[0, 0, 1, 0.96])

fig.tight_layout()

for stem in ("figA3_hs_probe_extra",):
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=150, bbox_inches="tight")
    print(f"Saved {OUT_DIR / stem}.pdf")
