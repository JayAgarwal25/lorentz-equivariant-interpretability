#!/usr/bin/env python3
"""Figure 1: mean relative logit change under pure boosts versus gamma for all five models."""

from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
EQ = ROOT / "results" / "equivariance_gamma5"
OUT = ROOT / "figures" / "main"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("lgatr",                        "L-GATr",      "#409107", "-"),
    ("lgatr_slim",                   "L-GATr-slim", "#1d828b", "--"),
    ("lloca_transformer_full_event", "LLoCa-T",     "#8b261d", "-."),
    ("vanilla_transformer",          "Vanilla-T",   "#e16c5b", ":"),
    ("part",                         "ParT",        "#e8c469", "--"),
]

MAX_GAMMA = 5.0
PLOT_FLOOR = 3e-4

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.2,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_ci(model_key: str) -> dict:
    f = np.load(EQ / model_key / "combined_bootstrap_ci.npz")
    gammas = f["gammas"] if "gammas" in f else None
    if gammas is None:
        for sd in sorted((EQ / model_key).iterdir()):
            if sd.is_dir() and sd.name.startswith("seed"):
                npz = list(sd.glob("*.npz"))
                if npz:
                    g = np.load(npz[0])
                    if "gammas" in g:
                        gammas = g["gammas"]
                        break
    return {
        "sweep_median": f["sweep_logit__median"],
        "sweep_lo":     f["sweep_logit__lo"],
        "sweep_hi":     f["sweep_logit__hi"],
        "gammas":       gammas,
    }


fig, ax = plt.subplots(figsize=(5.5, 3.5))

for key, label, color, ls in MODELS:
    ci = load_ci(key)
    gammas = ci["gammas"]
    if gammas is None:
        continue

    mask = gammas <= MAX_GAMMA
    g   = gammas[mask]
    med = ci["sweep_median"][mask]
    lo  = ci["sweep_lo"][mask]
    hi  = ci["sweep_hi"][mask]

    CLIP = 1e3
    med = np.clip(med, PLOT_FLOOR, CLIP)
    lo  = np.clip(lo,  PLOT_FLOOR, CLIP)
    hi  = np.clip(hi,  PLOT_FLOOR, CLIP)

    ax.fill_between(g, lo, hi, color=color, alpha=0.18, zorder=1)
    ax.plot(g, med, color=color, lw=2.0, ls=ls,
            marker="o", ms=4.5, label=label, zorder=3)

ax.set_yscale("log")
gamma_ticks = np.array([1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0])
ax.set_xticks(gamma_ticks)
ax.set_xticklabels(["1", "1.25", "1.5", "1.75", "2", "2.5", "3", "4", "5"])
ax.set_xlim(0.95, 5.15)
ax.set_ylim(3e-4, 2e1)
ax.set_xlabel(r"Boost factor $\gamma$")
ax.set_ylabel("Mean relative logit change")
ax.axhline(1e-2, color="gray", lw=0.8, ls=":", alpha=0.5)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=7.5,
           framealpha=0.9, bbox_to_anchor=(0.5, 1.02),
           handlelength=2.5, handletextpad=0.4, columnspacing=0.8)

fig.tight_layout(rect=[0, 0, 1, 0.96])


fig.savefig(OUT / "fig1_equivariance.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig1_equivariance.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUT}/fig1_equivariance.{{pdf,png}}")
