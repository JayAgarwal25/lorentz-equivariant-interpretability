#!/usr/bin/env python
"""Fit linear probes on the saved h_s of L-GATr and L-GATr-slim, per layer and target.

Regression targets use Ridge (alpha = 1) and R^2; top/QCD uses logistic regression and AUC; the
pT-rank quartile uses logistic regression and accuracy. Train and test jets are split 80/20,
particle targets follow their jet.

    python src/lgatr/fit_probes.py --lgatr results/probes/lgatr/seed1001/probe_data_lgatr.npz \\
        --slim results/probes/lgatr_slim/seed1001/probe_data_slim.npz \\
        --lgatr-output results/probes/lgatr/seed1001/probe_results \\
        --slim-output results/probes/lgatr_slim/seed1001/probe_results --output results/probes/lgatr_family/seed1001
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.linear_model    import Ridge, LogisticRegression
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import r2_score, roc_auc_score, accuracy_score
from sklearn.pipeline        import Pipeline

plt.rcParams.update({"text.usetex": False, "font.family": "sans-serif"})

# ──────────────────────────────────────────────────────────────────────────────
# Probe target definitions
# ──────────────────────────────────────────────────────────────────────────────

# (name, kind, level, label_key, display_title)
#   kind  : "regression" | "binary_cls" | "multi_cls"
#   level : "jet" | "particle"
PROBES = [
    # ── task ──────────────────────────────────────────────────────────────────
    ("top_qcd",        "binary_cls",  "jet",      "jet_labels",               "Top / QCD Classification"),
    # ── jet-level physics ─────────────────────────────────────────────────────
    ("jet_mass",       "regression",  "jet",      "jet_mass",                 "Jet Mass"),
    ("jet_mult",       "regression",  "jet",      "jet_multiplicity",         "Jet Multiplicity"),
    ("jet_tau1",       "regression",  "jet",      "jet_tau1",                 r"N-subjettiness $\tau_1$"),
    ("jet_tau2",       "regression",  "jet",      "jet_tau2",                 r"N-subjettiness $\tau_2$"),
    ("jet_tau3",       "regression",  "jet",      "jet_tau3",                 r"N-subjettiness $\tau_3$"),
    ("jet_tau21",      "regression",  "jet",      "jet_tau21",                r"N-subjettiness $\tau_2/\tau_1$"),
    ("jet_tau32",      "regression",  "jet",      "jet_tau32",                r"N-subjettiness $\tau_3/\tau_2$"),
    # ── particle-level input survival ─────────────────────────────────────────
    ("part_pt",        "regression",  "particle", "particle_pt",              "Particle $p_T$"),
    ("part_eta",       "regression",  "particle", "particle_eta",             r"Particle $\eta$"),
    ("part_phi",       "regression",  "particle", "particle_phi",             r"Particle $\phi$"),
    ("part_E",         "regression",  "particle", "particle_E",               "Particle Energy $E$"),
    # ── particle-level relational ──────────────────────────────────────────────
    ("part_quartile",  "multi_cls",   "particle", "particle_pt_rank_quartile","$p_T$ Rank Quartile"),
    ("part_dR",        "regression",  "particle", "particle_dR_to_axis",      r"$\Delta R$ to Jet Axis"),
]

# max number of particles to use for fitting (speed vs. memory trade-off)
MAX_PARTICLES_FIT = 150_000

# ──────────────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_npz(path):
    d = np.load(path)
    data = {k: d[k] for k in d.files}
    if "jet_tau21" not in data and {"jet_tau1", "jet_tau2"} <= data.keys():
        data["jet_tau21"] = data["jet_tau2"] / (data["jet_tau1"] + 1e-10)
    if "jet_tau32" not in data and {"jet_tau2", "jet_tau3"} <= data.keys():
        data["jet_tau32"] = data["jet_tau3"] / (data["jet_tau2"] + 1e-10)
    return data


def jet_train_test_split(n_jets, train_frac=0.8, seed=42):
    rng  = np.random.default_rng(seed)
    idx  = rng.permutation(n_jets)
    cut  = int(n_jets * train_frac)
    return idx[:cut], idx[cut:]


def particle_split(jet_train_idx, jet_test_idx, part_jet_idx):
    """Return particle indices that belong to train / test jets."""
    train_set = set(jet_train_idx.tolist())
    test_set  = set(jet_test_idx.tolist())
    all_idx   = np.arange(len(part_jet_idx))
    tr  = all_idx[np.isin(part_jet_idx, list(train_set))]
    te  = all_idx[np.isin(part_jet_idx, list(test_set))]
    return tr, te


def subsample(idx, max_n, seed=0):
    if len(idx) <= max_n:
        return idx
    rng = np.random.default_rng(seed)
    return rng.choice(idx, size=max_n, replace=False)

# ──────────────────────────────────────────────────────────────────────────────
# Probe fitting
# ──────────────────────────────────────────────────────────────────────────────

def make_probe(kind):
    if kind == "regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Ridge(alpha=1.0)),
        ])
    else:
        # binary or multi-class logistic regression
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model",  LogisticRegression(
                max_iter=1000, solver="lbfgs", C=1.0, n_jobs=-1,
            )),
        ])


def score_probe(kind, y_true, y_pred_or_proba):
    if kind == "regression":
        return r2_score(y_true, y_pred_or_proba)
    elif kind == "binary_cls":
        return roc_auc_score(y_true, y_pred_or_proba[:, 1])
    else:  # multi_cls
        return accuracy_score(y_true, y_pred_or_proba)


def fit_all_probes(data, model_name):
    """
    For each probe target and each layer, fit a linear probe and return scores.

    Returns
    -------
    results : dict[probe_name → np.ndarray of shape (N_LAYERS,)]
    """
    n_jets   = data["jet_reps"].shape[1]
    n_layers = data["jet_reps"].shape[0]

    jtr, jte = jet_train_test_split(n_jets)
    part_jet = data["particle_jet_idx"]
    ptr, pte = particle_split(jtr, jte, part_jet)
    ptr_sub  = subsample(ptr, MAX_PARTICLES_FIT)

    results = {}

    for pname, kind, level, label_key, title in PROBES:
        print(f"  [{model_name}]  {pname:20s}  ({level}, {kind})")
        scores = np.full(n_layers, np.nan)

        y_all = data[label_key]

        for li in range(n_layers):
            if level == "jet":
                X_all  = data["jet_reps"][li]          # (n_jets, s_ch)
                X_tr   = X_all[jtr];  y_tr = y_all[jtr]
                X_te   = X_all[jte];  y_te = y_all[jte]
            else:
                X_all  = data["particle_reps"][li]     # (total_p, s_ch)
                X_tr   = X_all[ptr_sub];  y_tr = y_all[ptr_sub]
                X_te   = X_all[pte];      y_te = y_all[pte]

            probe = make_probe(kind)
            probe.fit(X_tr, y_tr)

            if kind == "regression":
                y_hat    = probe.predict(X_te)
                scores[li] = score_probe(kind, y_te, y_hat)
            elif kind == "binary_cls":
                y_proba  = probe.predict_proba(X_te)
                scores[li] = score_probe(kind, y_te, y_proba)
            else:  # multi_cls
                y_hat    = probe.predict(X_te)
                scores[li] = score_probe(kind, y_te, y_hat)

        results[pname] = scores
        layer_str = "  ".join([f"L{i}={s:.3f}" for i, s in enumerate(scores)])
        print(f"    {layer_str}")

    return results

# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

COLORS = {
    "lgatr": "#1f77b4",   # blue
    "slim":  "#d62728",   # red
}

METRIC_LABEL = {
    "regression":  "R²",
    "binary_cls":  "AUC",
    "multi_cls":   "Accuracy",
}


def plot_probe_curves(results_lgatr, results_slim, output_dir):
    """
    For each probe: one figure with both models overlaid.
    Also one summary figure showing all probes on a grid.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    n_layers = 13
    x        = np.arange(n_layers)
    x_labels = ["lin_in"] + [str(i) for i in range(1, n_layers)]

    # ── per-probe individual plots ────────────────────────────────────────────
    for pname, kind, level, label_key, title in PROBES:
        fig, ax = plt.subplots(figsize=(8, 4))

        for mname, res in [("L-GATr", results_lgatr), ("L-GATr-slim", results_slim)]:
            color = COLORS["lgatr"] if "slim" not in mname else COLORS["slim"]
            ax.plot(x, res[pname], "o-", color=color, label=mname, linewidth=2, markersize=5)

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_xlabel("Layer (0 = after linear_in)", fontsize=12)
        ax.set_ylabel(METRIC_LABEL[kind], fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        if kind == "regression":
            ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
        elif kind == "binary_cls":
            ax.set_ylim([0.5, 1.0])
        else:
            ax.set_ylim([0.0, 1.0])

        fig.tight_layout()
        fig.savefig(out / f"probe_{pname}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # ── summary grid (all probes, 4 columns) ─────────────────────────────────
    n_probes = len(PROBES)
    ncols    = 4
    nrows    = (n_probes + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes_flat = axes.flatten()

    for pi, (pname, kind, level, label_key, title) in enumerate(PROBES):
        ax = axes_flat[pi]
        for mname, res in [("L-GATr", results_lgatr), ("L-GATr-slim", results_slim)]:
            color = COLORS["lgatr"] if "slim" not in mname else COLORS["slim"]
            ax.plot(x, res[pname], "o-", color=color, label=mname,
                    linewidth=1.5, markersize=4)
        ax.set_xticks(x[::2])
        ax.set_xticklabels(x_labels[::2], fontsize=7)
        ax.set_ylabel(METRIC_LABEL[kind], fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.3)
        if kind == "regression":
            ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))

    # Hide unused subplots
    for pi in range(n_probes, len(axes_flat)):
        axes_flat[pi].set_visible(False)

    # Single legend for entire figure
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=11, ncol=2)

    fig.suptitle(
        "Linear Probe Results — L-GATr vs L-GATr-slim\nPer-layer R² / AUC / Accuracy",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out / "probe_summary_grid.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved summary grid → {out / 'probe_summary_grid.png'}")

    # ── grouped comparison: jet-level vs particle-level ───────────────────────
    for group_name, group_probes in [
        ("jet_level",      [p for p in PROBES if p[2] == "jet"]),
        ("particle_level", [p for p in PROBES if p[2] == "particle"]),
    ]:
        ng    = len(group_probes)
        ncols = min(4, ng)
        nrows = (ng + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows),
                                 squeeze=False)
        axes_flat = axes.flatten()

        for pi, (pname, kind, level, label_key, title) in enumerate(group_probes):
            ax = axes_flat[pi]
            for mname, res in [("L-GATr", results_lgatr), ("L-GATr-slim", results_slim)]:
                color = COLORS["lgatr"] if "slim" not in mname else COLORS["slim"]
                ax.plot(x, res[pname], "o-", color=color, label=mname,
                        linewidth=2, markersize=5)
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, fontsize=7)
            ax.set_xlabel("Layer", fontsize=9)
            ax.set_ylabel(METRIC_LABEL[kind], fontsize=9)
            ax.set_title(title, fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            if kind == "regression":
                ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))

        for pi in range(ng, len(axes_flat)):
            axes_flat[pi].set_visible(False)

        fig.suptitle(
            f"Linear Probes — {'Jet' if group_name == 'jet_level' else 'Particle'}-level targets",
            fontsize=13,
        )
        fig.tight_layout()
        fig.savefig(out / f"probe_{group_name}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def save_scores_table(results_lgatr, results_slim, output_dir):
    """Save a plain-text table of final-layer scores for easy reference."""
    out = Path(output_dir) / "probe_scores_final_layer.txt"
    lines = [
        f"{'Probe':<28}  {'Kind':<12}  {'Level':<10}  "
        f"{'LGATr L12':>10}  {'Slim L12':>10}",
        "-" * 80,
    ]
    for pname, kind, level, label_key, title in PROBES:
        s_lg  = results_lgatr[pname][-1]
        s_sl  = results_slim[pname][-1]
        metric = METRIC_LABEL[kind]
        lines.append(
            f"{pname:<28}  {kind:<12}  {level:<10}  "
            f"{s_lg:>10.4f}  {s_sl:>10.4f}  ({metric})"
        )
    text = "\n".join(lines)
    out.write_text(text)
    print(f"\nScores table → {out}")
    print(text)


def save_scores_npz(results_lgatr, results_slim, output_dir):
    """Save all per-layer scores as .npz for later analysis."""
    payload = {}
    for pname, *_ in PROBES:
        payload[f"lgatr_{pname}"] = results_lgatr[pname]
        payload[f"slim_{pname}"]  = results_slim[pname]
    np.savez(Path(output_dir) / "probe_scores.npz", **payload)
    print(f"Scores npz → {Path(output_dir) / 'probe_scores.npz'}")


def save_single_model_scores(results, output_dir, model_label):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {pname: results[pname] for pname, *_ in PROBES}
    np.savez(out / "probe_scores.npz", **payload)

    lines = [
        f"{'Probe':<28}  {'Kind':<12}  {'Level':<10}  {model_label:>10}",
        "-" * 70,
    ]
    for pname, kind, level, label_key, title in PROBES:
        metric = METRIC_LABEL[kind]
        lines.append(
            f"{pname:<28}  {kind:<12}  {level:<10}  {results[pname][-1]:>10.4f}  ({metric})"
        )
    (out / "probe_scores_final_layer.txt").write_text("\n".join(lines))
    print(f"Per-model scores saved → {out}")

# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lgatr",  required=True, help="Path to probe_data_lgatr.npz")
    ap.add_argument("--slim",   required=True, help="Path to probe_data_slim.npz")
    ap.add_argument("--output", default="results/probe_results/")
    ap.add_argument("--lgatr-output", default=None, help="Optional per-model output dir for LGATr scores.")
    ap.add_argument("--slim-output", default=None, help="Optional per-model output dir for L-GATr-slim scores.")
    args = ap.parse_args()

    print("Loading data...")
    data_lgatr = load_npz(args.lgatr)
    data_slim  = load_npz(args.slim)

    print(f"\nLGATr  — jet_reps: {data_lgatr['jet_reps'].shape}  "
          f"particle_reps: {data_lgatr['particle_reps'].shape}")
    print(f"Slim   — jet_reps: {data_slim['jet_reps'].shape}  "
          f"particle_reps: {data_slim['particle_reps'].shape}")

    print("\n── Fitting probes (L-GATr) ─────────────────────────────────────────")
    results_lgatr = fit_all_probes(data_lgatr, "lgatr")

    print("\n── Fitting probes (L-GATr-slim) ────────────────────────────────────")
    results_slim  = fit_all_probes(data_slim,  "slim")

    print("\n── Plotting ────────────────────────────────────────────────────────")
    plot_probe_curves(results_lgatr, results_slim, args.output)
    save_scores_table(results_lgatr, results_slim, args.output)
    save_scores_npz(results_lgatr, results_slim, args.output)
    if args.lgatr_output:
        save_single_model_scores(results_lgatr, args.lgatr_output, "LGATr")
    if args.slim_output:
        save_single_model_scores(results_slim, args.slim_output, "LGATr-slim")

    print(f"\nAll results saved to {args.output}")


if __name__ == "__main__":
    main()
