#!/usr/bin/env python3
"""Figure 14: channel ablations for L-GATr-slim."""

from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
ABL  = ROOT / "results" / "ablations"
OUT  = ROOT / "figures" / "appendix"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = ["seed1001", "seed1002", "seed1003"]

# order matches abl_names in grade_decomp_slim.npz
ABLATIONS = [
    ("zero_v",          r"zero $h_v$",       "#1f77b4"),
    ("zero_s",          r"zero $h_s$",        "#ff7f0e"),
    ("zero_t",          r"zero $e_t$",        "#2ca02c"),
    ("zero_spatial",    r"zero $e_{xyz}$",    "#8c564b"),
    ("zero_z",          r"zero $e_z$",        "#9467bd"),
    ("zero_transverse", r"zero $e_{xy}$",     "#e377c2"),
]

# Keys in abl_names that have layer-resolved data
LAYER_KEYS = ["zero_v", "zero_s"]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.alpha": 0.2, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
})

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(8.5, 3.2),
                                   gridspec_kw={"width_ratios": [1.3, 1.6]})

# ── left panel: global zero-ablation ─────────────────────────────────────────
ci = np.load(ABL / "lgatr_slim" / "combined_bootstrap_ci.npz")

# Build per-seed ΔAUC lookup: key → [s1, s2, s3]
seed_vals = {key: [] for key, _, _ in ABLATIONS}
for seed in SEEDS:
    f = np.load(ABL / "lgatr_slim" / seed / "grade_decomp_slim.npz", allow_pickle=True)
    baseline = float(f["baseline_auc"])
    abl_names = list(f["abl_names"])
    abl_aucs  = f["abl_aucs"]
    for key, _, _ in ABLATIONS:
        idx = abl_names.index(key)
        seed_vals[key].append(baseline - float(abl_aucs[idx]))

x_pos = np.arange(len(ABLATIONS))

for xi, (key, label, color) in enumerate(ABLATIONS):
    med = float(ci[f"{key}__median"])
    lo  = float(ci[f"{key}__lo"])
    hi  = float(ci[f"{key}__hi"])

    ax_l.bar(xi, med, width=0.5, color=color, alpha=0.75, zorder=2)
    ax_l.errorbar(xi, med, yerr=[[med - lo], [hi - med]],
                  fmt="none", color="black", capsize=4, lw=1.5, zorder=4)
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.08, 0.08, len(SEEDS))
    for sv, jit in zip(seed_vals[key], jitter):
        ax_l.plot(xi + jit, sv, "o", color=color, ms=5.5,
                  markeredgecolor="white", markeredgewidth=0.6,
                  alpha=0.9, zorder=5)

ax_l.set_xticks(x_pos)
ax_l.set_xticklabels([a[1] for a in ABLATIONS], fontsize=8)
ax_l.set_ylabel(r"$\Delta\mathrm{AUC}$ (zero-channel ablation)")
ax_l.set_ylim(-0.02, 0.85)
ax_l.axhline(0, color="gray", lw=0.7, ls="--", alpha=0.5)

# ── right panel: layer-resolved (zero_v and zero_s only) ─────────────────────
n_layers = 13
layer_data = {key: np.zeros((len(SEEDS), n_layers)) for key in LAYER_KEYS}

for si, seed in enumerate(SEEDS):
    f = np.load(ABL / "lgatr_slim_layer" / seed / "grade_decomp_slim_layer.npz",
                allow_pickle=True)
    baseline  = float(f["baseline_auc"])
    abl_names = list(f["abl_names"])
    layer_aucs = np.array(f["layer_aucs"], dtype=float)  # (2, 13)
    for key in LAYER_KEYS:
        idx = abl_names.index(key)
        layer_data[key][si] = baseline - layer_aucs[idx]

layers = np.arange(n_layers)
seed_styles = ["--", "-.", ":"]

color_map = {a[0]: (a[1], a[2]) for a in ABLATIONS}
for key in LAYER_KEYS:
    label, color = color_map[key]
    arr  = layer_data[key]
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0)
    ax_r.plot(layers, mean, color=color, lw=2.2, label=label, zorder=3)
    ax_r.fill_between(layers, mean - std, mean + std,
                      color=color, alpha=0.18, zorder=2)

# per-seed thin lines for zero_v (more variable)
for si, sty in enumerate(seed_styles):
    ax_r.plot(layers, layer_data["zero_v"][si],
              color=color_map["zero_v"][1], lw=0.9, ls=sty, alpha=0.55, zorder=2)

ax_r.axhline(0, color="gray", lw=0.7, ls="--", alpha=0.5)
ax_r.set_xlabel("Layer")
ax_r.set_ylabel(r"$\Delta\mathrm{AUC}$ (zero-channel ablation)")
ax_r.set_xticks(range(0, n_layers, 2))
ax_r.set_ylim(-0.05, 0.85)
ax_r.legend(frameon=False, loc="upper center")

fig.tight_layout(pad=0.5, w_pad=1.2)
fig.savefig(OUT / "figA5b_slim.pdf", bbox_inches="tight")
fig.savefig(OUT / "figA5b_slim.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUT}/figA_slim.{{pdf,png}}")
