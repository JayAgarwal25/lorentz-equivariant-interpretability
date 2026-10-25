#!/usr/bin/env python3
"""Figure 15: grade ablations for L-GATr with five independent grades."""

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

# grade_aucs ordering matches grade_names: G0, G1, G2, G3, G4
GROUPS = [
    ("g0", "G0\n(scalar)",      "#ff7f0e"),
    ("g1", "G1\n(vector)",      "#1f77b4"),
    ("g2", "G2\n(bivector)",    "#d62728"),
    ("g3", "G3\n(trivector)",   "#9467bd"),
    ("g4", "G4\n(pseudoscalar)","#8c564b"),
]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.alpha": 0.2, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
})

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(8.5, 3.2),
                                   gridspec_kw={"width_ratios": [1.2, 1.6]})

# ── left panel: global zero-ablation ─────────────────────────────────────────
ci = np.load(ABL / "lgatr_subgroup_false" / "combined_bootstrap_ci.npz")

seed_vals = {g: [] for g, _, _ in GROUPS}
for seed in SEEDS:
    f = np.load(ABL / "lgatr_subgroup_false" / seed / "grade_decomp_full.npz",
                allow_pickle=True)
    baseline = float(f["baseline_auc"])
    for gi, (gkey, _, _) in enumerate(GROUPS):
        seed_vals[gkey].append(baseline - float(f["grade_aucs"][gi]))

x_pos = np.arange(len(GROUPS))

for xi, (gkey, glabel, color) in enumerate(GROUPS):
    med = float(ci[f"zero_{gkey}__median"])
    lo  = float(ci[f"zero_{gkey}__lo"])
    hi  = float(ci[f"zero_{gkey}__hi"])

    ax_l.bar(xi, med, width=0.5, color=color, alpha=0.75, zorder=2)
    ax_l.errorbar(xi, med, yerr=[[med - lo], [hi - med]],
                  fmt="none", color="black", capsize=4, lw=1.5, zorder=4)
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.08, 0.08, len(SEEDS))
    for sv, jit in zip(seed_vals[gkey], jitter):
        ax_l.plot(xi + jit, sv, "o", color=color, ms=5.5,
                  markeredgecolor="white", markeredgewidth=0.6,
                  alpha=0.9, zorder=5)

ax_l.set_xticks(x_pos)
ax_l.set_xticklabels([g[1] for g in GROUPS])
ax_l.set_ylabel(r"$\Delta\mathrm{AUC}$ (zero-grade ablation)")
ax_l.set_ylim(-0.02, 0.6)
ax_l.axhline(0, color="gray", lw=0.7, ls="--", alpha=0.5)

# ── right panel: layer-resolved ───────────────────────────────────────────────
n_layers = 13
layer_data = {gkey: np.zeros((len(SEEDS), n_layers)) for gkey, _, _ in GROUPS}

for si, seed in enumerate(SEEDS):
    f = np.load(ABL / "lgatr_subgroup_false_layer" / seed / "grade_decomp_layer.npz",
                allow_pickle=True)
    baseline = float(f["baseline_auc"])
    layer_aucs = np.array(f["layer_aucs"], dtype=float)  # (5, 13)
    for gi, (gkey, _, _) in enumerate(GROUPS):
        layer_data[gkey][si] = baseline - layer_aucs[gi]

layers = np.arange(n_layers)
seed_styles = ["--", "-.", ":"]

for gkey, glabel, color in GROUPS:
    arr  = layer_data[gkey]
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0)
    ax_r.plot(layers, mean, color=color, lw=2.2,
              label=glabel.replace("\n", " "), zorder=3)
    ax_r.fill_between(layers, mean - std, mean + std,
                      color=color, alpha=0.18, zorder=2)

# per-seed thin lines for G1 (dominant + most variable)
for si, sty in enumerate(seed_styles):
    ax_r.plot(layers, layer_data["g1"][si],
              color=GROUPS[1][2], lw=0.9, ls=sty, alpha=0.55, zorder=2)

ax_r.axhline(0, color="gray", lw=0.7, ls="--", alpha=0.5)
ax_r.set_xlabel("Layer")
ax_r.set_ylabel(r"$\Delta\mathrm{AUC}$ (zero-grade ablation)")
ax_r.set_xticks(range(0, n_layers, 2))
ax_r.set_ylim(-0.05, 0.6)
ax_r.legend(frameon=False, loc="upper right", ncol=2)

ax_r.annotate(r"All grades $\to 0$ at L12" + "\n(architectural constraint)",
              xy=(12, 0), xytext=(8.5, 0.18),
              arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
              fontsize=7.5, color="gray", ha="center")

fig.tight_layout(pad=0.5, w_pad=1.2)
fig.savefig(OUT / "figA5c_subgroup_false.pdf", bbox_inches="tight")
fig.savefig(OUT / "figA5c_subgroup_false.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUT}/figA5c_subgroup_false.{{pdf,png}}")
