#!/usr/bin/env python3
"""Figure 2: final-layer linear probe scores for selected targets across all five models."""

from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT   = Path(__file__).parent.parent
PROBES = ROOT / "results" / "probes"
OUT    = ROOT / "figures" / "main"
OUT.mkdir(parents=True, exist_ok=True)

# most → least physics-informed
MODELS = [
    ("lgatr",               "L-GATr",      "#409107", "o",  7.0),
    ("lgatr_slim",          "L-GATr-slim", "#1d828b", "s",  6.5),
    ("lloca_transformer",   "LLoCa-T",     "#8b261d", "^",  7.0),
    ("part",                "ParT",        "#e8c469", "D",  5.5),
    ("vanilla_transformer", "Vanilla-T",   "#e16c5b", "v",  6.5),
]

# Ordered by semantic blocks, with the more discriminating blocks first, after
# dropping top/QCD and particle pT-rank quartile from the main-text summary.
PROBES_MAIN = [
    ("part_eta",  r"$\eta$"),
    ("jet_tau1",  r"$\tau_1$"),
    ("jet_tau2",  r"$\tau_2$"),
    ("jet_tau3",  r"$\tau_3$"),
    ("jet_tau21", r"$\tau_{21}$"),
    ("jet_tau32", r"$\tau_{32}$"),
    ("jet_mass",  "mass"),
    ("jet_mult",  "mult"),
    ("part_pt",   r"$p_T$"),
    ("part_E",    r"$E$"),
    ("part_dR",   r"$\Delta R$"),
    ("part_phi",  r"$\phi$"),
]

N_MODELS = len(MODELS)
offsets  = np.linspace(-0.30, 0.30, N_MODELS)

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


def draw_panel(ax, probe_list):
    for mi, (model_key, label, color, marker, ms) in enumerate(MODELS):
        ci = np.load(PROBES / model_key / "combined_bootstrap_ci.npz")
        xs, meds, los, his = [], [], [], []
        for pi, (target, _) in enumerate(probe_list):
            med = float(ci[f"{target}__median"][-1])
            lo  = float(ci[f"{target}__lo"][-1])
            hi  = float(ci[f"{target}__hi"][-1])
            xs.append(pi + offsets[mi])
            meds.append(med)
            los.append(med - lo)
            his.append(hi - med)
        xs = np.array(xs)
        meds = np.array(meds)
        ax.errorbar(xs, meds, yerr=[los, his],
                    fmt="none", color=color, lw=1.0, capsize=2.5, alpha=0.7, zorder=2)
        ax.plot(xs, meds, marker=marker, color=color, ms=ms, lw=0,
                label=label, markeredgecolor="white", markeredgewidth=0.5, zorder=3)

    ax.set_xticks(np.arange(len(probe_list)))
    ax.set_xticklabels([p[1] for p in probe_list])
    ax.set_xlim(-0.65, len(probe_list) - 0.35)
    ax.set_ylim(-0.08, 1.08)
    ax.set_ylabel(r"Final-layer $R^2$")
    ax.axhline(0, color="gray", lw=0.7, ls="--", alpha=0.4)


fig, ax = plt.subplots(1, 1, figsize=(10.5, 3.0))

draw_panel(ax, PROBES_MAIN)

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=8,
           framealpha=0.9, bbox_to_anchor=(0.5, 1.01),
           handlelength=1.0, handletextpad=0.4, columnspacing=0.8)

fig.tight_layout(rect=[0, 0, 1, 0.91])

fig.savefig(OUT / "fig3_probe_bar.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig3_probe_bar.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUT}/fig3_probe_bar.{{pdf,png}}")
