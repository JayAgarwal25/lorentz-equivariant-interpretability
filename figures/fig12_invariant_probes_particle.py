#!/usr/bin/env python3
"""Figure 12: geometric-algebra-invariant probe trajectories for the particle-level targets, by grade group."""

from __future__ import annotations
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT   = Path(__file__).parent.parent
PROBES = ROOT / "results" / "probes" / "lgatr"
OUT_DIR = ROOT / "figures" / "appendix"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS    = ["seed1001", "seed1002", "seed1003"]
N_LAYERS = 13

GROUPS = [
    ("scalar_like",   "Scalar-like (G0+G4)", "#ff7f0e", "-",  2.0),
    ("vector_like",   "Vector-like (G1+G3)", "#1f77b4", "--", 2.0),
    ("bivector_like", "Bivector (G2)",        "#d62728", ":",  2.0),
    ("All",           "All (832 features)",   "#000000", "-",  1.8),
    ("h_s",           r"$h_s$ (scalar ch.)",  "#8c564b", "-.", 1.8),
]

TARGETS = [
    ("part_pt",       r"$R^2$ (particle $p_T$)"),
    ("part_eta",      r"$R^2$ (particle $\eta$)"),
    ("part_phi",      r"$R^2$ (particle $\phi$)"),
    ("part_E",        r"$R^2$ (particle $E$)"),
    ("part_quartile", r"Acc. (particle $p_T$-quartile)"),
    ("part_dR",       r"$R^2$ (particle $\Delta R$)"),
]

ALL_GROUPS = [g for g, *_ in GROUPS]


def load_seed_data():
    all_targets = [t for t, _ in TARGETS]
    data = {t: {g: [] for g in ALL_GROUPS} for t in all_targets}
    for seed in SEEDS:
        path = PROBES / seed / "mv_probe_results" / "results.json"
        with open(path) as f:
            res = json.load(f)
        for li in range(N_LAYERS):
            layer_data = res[f"L{li}"]
            for tgt in all_targets:
                for g in ALL_GROUPS:
                    val = layer_data.get(tgt, {}).get(g, float("nan"))
                    data[tgt][g].append(val)
    result = {}
    for tgt in all_targets:
        result[tgt] = {}
        for g in ALL_GROUPS:
            arr = np.array(data[tgt][g])
            result[tgt][g] = arr.reshape(len(SEEDS), N_LAYERS) if arr.size == len(SEEDS) * N_LAYERS \
                             else np.full((len(SEEDS), N_LAYERS), np.nan)
    return result


seed_data = load_seed_data()
x = np.arange(N_LAYERS)

n_rows, n_cols = 2, 3
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 7.5))

handles_for_legend = None

for idx, (target, ylabel) in enumerate(TARGETS):
    ax = axes[idx // n_cols, idx % n_cols]

    for group_key, group_label, color, ls, lw in GROUPS:
        scores = seed_data[target][group_key]
        mean = np.nanmean(scores, axis=0)
        std  = np.nanstd(scores, axis=0)
        line, = ax.plot(x, mean, color=color, lw=lw, ls=ls, label=group_label, zorder=3)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12, zorder=2)

    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(range(0, N_LAYERS, 2))
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3)

    if idx == 0:
        handles_for_legend, labels_for_legend = ax.get_legend_handles_labels()

fig.legend(handles_for_legend, labels_for_legend, loc="upper center", ncol=5,
           fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=[0, 0, 1, 0.96])

fig.savefig(OUT_DIR / "figA4b_mv_probe_particle.pdf", bbox_inches="tight")
fig.savefig(OUT_DIR / "figA4b_mv_probe_particle.png", dpi=150, bbox_inches="tight")
print(f"Saved {OUT_DIR}/figA4b_mv_probe_particle.pdf")
