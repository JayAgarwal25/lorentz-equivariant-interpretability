#!/usr/bin/env python3
"""
Fit linear probes on ParT representations and plot per-layer scores.
Optionally overlays with LGATr-full and/or LGATr-slim (TopTagging) for comparison.

Usage:
  cd path/to/particle_transformer
  python3.10 fit_probes_part.py \\
      --part    results/part_interp/probe_data_part.npz \\
      --lgatr   ../lorentz-gatr/results/probe_data_lgatr.npz \\
      --slim    ../tagger-quantization/results/probe_data_slim.npz \\
      --output  results/part_interp/probe_results/
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.linear_model  import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import r2_score, roc_auc_score, accuracy_score
from sklearn.pipeline      import Pipeline

plt.rcParams.update({"text.usetex": False, "font.family": "sans-serif"})

PROBES = [
    ("top_qcd",        "binary_cls",  "jet",      "jet_labels",               "Top / QCD Classification"),
    ("jet_mass",       "regression",  "jet",      "jet_mass",                 "Jet Mass"),
    ("jet_mult",       "regression",  "jet",      "jet_multiplicity",         "Jet Multiplicity"),
    ("jet_tau1",       "regression",  "jet",      "jet_tau1",                 r"$\tau_1$"),
    ("jet_tau2",       "regression",  "jet",      "jet_tau2",                 r"$\tau_2$"),
    ("jet_tau3",       "regression",  "jet",      "jet_tau3",                 r"$\tau_3$"),
    ("jet_tau21",      "regression",  "jet",      "jet_tau21",                r"$\tau_2/\tau_1$"),
    ("jet_tau32",      "regression",  "jet",      "jet_tau32",                r"$\tau_3/\tau_2$"),
    ("part_pt",        "regression",  "particle", "particle_pt",              r"Particle $p_T$"),
    ("part_eta",       "regression",  "particle", "particle_eta",             r"Particle $\eta$"),
    ("part_phi",       "regression",  "particle", "particle_phi",             r"Particle $\phi$"),
    ("part_E",         "regression",  "particle", "particle_E",               "Particle $E$"),
    ("part_quartile",  "multi_cls",   "particle", "particle_pt_rank_quartile", r"$p_T$ Rank Quartile"),
    ("part_dR",        "regression",  "particle", "particle_dR_to_axis",      r"$\Delta R$ to Jet Axis"),
]

MAX_PARTICLES_FIT = 150_000
METRIC_LABEL = {"regression": "R²", "binary_cls": "AUC", "multi_cls": "Accuracy"}

COLORS = {
    "part":  "#2ca02c",   # green  — ParT
    "lgatr": "#1f77b4",   # blue   — LGATr-full
    "slim":  "#d62728",   # red    — LGATr-slim
}
LABELS = {
    "part":  "ParT (scratch)",
    "lgatr": "LGATr-full",
    "slim":  "LGATr-slim",
}


def load_npz(path):
    d = np.load(path)
    data = {k: d[k] for k in d.files}
    if "jet_tau21" not in data and {"jet_tau1", "jet_tau2"} <= data.keys():
        data["jet_tau21"] = data["jet_tau2"] / (data["jet_tau1"] + 1e-10)
    if "jet_tau32" not in data and {"jet_tau2", "jet_tau3"} <= data.keys():
        data["jet_tau32"] = data["jet_tau3"] / (data["jet_tau2"] + 1e-10)
    return data


def jet_train_test_split(n_jets, train_frac=0.8, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_jets)
    cut = int(n_jets * train_frac)
    return idx[:cut], idx[cut:]


def particle_split(jtr, jte, part_jet_idx):
    ts = set(jtr.tolist()); te_s = set(jte.tolist())
    all_idx = np.arange(len(part_jet_idx))
    return all_idx[np.isin(part_jet_idx, list(ts))], all_idx[np.isin(part_jet_idx, list(te_s))]


def subsample(idx, max_n, seed=0):
    if len(idx) <= max_n: return idx
    return np.random.default_rng(seed).choice(idx, size=max_n, replace=False)


def make_probe(kind):
    if kind == "regression":
        return Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
    return Pipeline([("scaler", StandardScaler()),
                     ("model", LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0, n_jobs=-1))])


def score(kind, y_te, probe, X_te):
    if kind == "regression":   return r2_score(y_te, probe.predict(X_te))
    if kind == "binary_cls":   return roc_auc_score(y_te, probe.predict_proba(X_te)[:, 1])
    return accuracy_score(y_te, probe.predict(X_te))


def fit_all(data, tag):
    n_jets   = data["jet_reps"].shape[1]
    n_layers = data["jet_reps"].shape[0]
    jtr, jte = jet_train_test_split(n_jets)
    ptr, pte = particle_split(jtr, jte, data["particle_jet_idx"])
    ptr_sub  = subsample(ptr, MAX_PARTICLES_FIT)
    results  = {}

    for pname, kind, level, label_key, title in PROBES:
        print(f"  [{tag}]  {pname:20s}  ({level}, {kind})")
        scores = np.full(n_layers, np.nan)
        y_all  = data[label_key]
        for li in range(n_layers):
            if level == "jet":
                X_tr, y_tr = data["jet_reps"][li][jtr], y_all[jtr]
                X_te, y_te = data["jet_reps"][li][jte], y_all[jte]
            else:
                X_tr, y_tr = data["particle_reps"][li][ptr_sub], y_all[ptr_sub]
                X_te, y_te = data["particle_reps"][li][pte],     y_all[pte]
            probe = make_probe(kind)
            probe.fit(X_tr, y_tr)
            scores[li] = score(kind, y_te, probe, X_te)
        results[pname] = scores
        print("    " + "  ".join(f"L{i}={s:.3f}" for i, s in enumerate(scores)))
    return results


def plot_curves(results_dict, n_layers_dict, output_dir):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)

    n_probes = len(PROBES); ncols = 4; nrows = (n_probes + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.5*nrows))
    axes_flat = axes.flatten()

    for pi, (pname, kind, level, label_key, title) in enumerate(PROBES):
        ax = axes_flat[pi]
        for tag, res in results_dict.items():
            if res is None or pname not in res: continue
            n = n_layers_dict[tag]
            x = np.arange(n)
            mk = {"part": "o-", "lgatr": "^-", "slim": "s--"}[tag]
            ax.plot(x, res[pname], mk, color=COLORS[tag], label=LABELS[tag],
                    linewidth=1.5, markersize=4)
        ax.set_ylabel(METRIC_LABEL[kind], fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.3)
        if kind == "regression": ax.set_ylim(bottom=min(0., ax.get_ylim()[0]))

    for pi in range(n_probes, len(axes_flat)):
        axes_flat[pi].set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=10, ncol=3)
    fig.suptitle("Linear Probe Results — ParT vs LGATr-full vs LGATr-slim (TopTagging)\nPer-layer R² / AUC / Accuracy",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out / "probe_summary_grid.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved summary grid → {out}/probe_summary_grid.png")

    # Individual plots
    for pname, kind, level, label_key, title in PROBES:
        fig, ax = plt.subplots(figsize=(8, 4))
        for tag, res in results_dict.items():
            if res is None or pname not in res: continue
            n = n_layers_dict[tag]
            x = np.arange(n)
            mk = {"part": "o-", "lgatr": "^-", "slim": "s--"}[tag]
            xl = (["embed"] + [str(i) for i in range(1, n)]) if tag == "part" else \
                 (["lin_in"] + [str(i) for i in range(1, n)])
            ax.plot(x, res[pname], mk, color=COLORS[tag], label=LABELS[tag],
                    linewidth=2, markersize=5)
            ax.set_xticks(x); ax.set_xticklabels(xl, fontsize=7)
        ax.set_xlabel("Layer", fontsize=12); ax.set_ylabel(METRIC_LABEL[kind], fontsize=12)
        ax.set_title(title, fontsize=13); ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
        if kind == "regression": ax.set_ylim(bottom=min(0., ax.get_ylim()[0]))
        fig.tight_layout()
        fig.savefig(out / f"probe_{pname}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def save_scores(results_dict, n_layers_dict, output_dir):
    out = Path(output_dir)
    tags = [t for t in ["part", "lgatr", "slim"] if results_dict.get(t) is not None]
    if tags == ["part"]:
        lines = [
            f"{'Probe':<25}  {'Kind':<12}  {'Level':<10}  {'ParT L'+str(n_layers_dict['part']-1):>14}"
        ]
        lines.append("-" * (len(lines[0]) + 5))
        for pname, kind, level, label_key, title in PROBES:
            score = results_dict["part"][pname][-1]
            lines.append(
                f"{pname:<25}  {kind:<12}  {level:<10}  {score:>14.4f}  ({METRIC_LABEL[kind]})"
            )
        text = "\n".join(lines)
        (out / "probe_scores_final_layer.txt").write_text(text)
        print(f"\n{text}")
        np.savez(out / "probe_scores.npz", **results_dict["part"])
        return

    header = f"{'Probe':<25}  {'Kind':<12}  {'Level':<10}" + \
             "".join(f"  {LABELS[t]+' L'+str(n_layers_dict[t]-1):>14}" for t in tags)
    lines = [header, "-" * (len(header) + 5)]
    for pname, kind, level, label_key, title in PROBES:
        row = f"{pname:<25}  {kind:<12}  {level:<10}"
        for t in tags:
            s = results_dict[t][pname][-1] if results_dict.get(t) and pname in results_dict[t] else float("nan")
            row += f"  {s:>14.4f}"
        row += f"  ({METRIC_LABEL[kind]})"
        lines.append(row)
    text = "\n".join(lines)
    (out / "probe_scores_final_layer.txt").write_text(text)
    print(f"\n{text}")

    payload = {}
    for t, res in results_dict.items():
        if res:
            payload.update({f"{t}_{p}": res[p] for p, *_ in PROBES if p in res})
    np.savez(out / "probe_scores.npz", **payload)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part",   required=True)
    ap.add_argument("--lgatr",  default=None)
    ap.add_argument("--slim",   default=None)
    ap.add_argument("--output", default="results/part_interp/probe_results/")
    args = ap.parse_args()

    results_dict  = {}
    n_layers_dict = {}

    print(f"Loading ParT probes from {args.part} ...")
    data_part = load_npz(args.part)
    n_layers_dict["part"] = data_part["jet_reps"].shape[0]
    print(f"  jet_reps: {data_part['jet_reps'].shape}  particle_reps: {data_part['particle_reps'].shape}")

    data_lgatr = None
    if args.lgatr:
        print(f"Loading LGATr-full probes from {args.lgatr} ...")
        data_lgatr = load_npz(args.lgatr)
        n_layers_dict["lgatr"] = data_lgatr["jet_reps"].shape[0]
        print(f"  jet_reps: {data_lgatr['jet_reps'].shape}  particle_reps: {data_lgatr['particle_reps'].shape}")

    data_slim = None
    if args.slim:
        print(f"Loading LGATr-slim probes from {args.slim} ...")
        data_slim = load_npz(args.slim)
        n_layers_dict["slim"] = data_slim["jet_reps"].shape[0]
        print(f"  jet_reps: {data_slim['jet_reps'].shape}  particle_reps: {data_slim['particle_reps'].shape}")

    print("\n── Fitting ParT probes ─────────────────────────────────────────")
    results_dict["part"] = fit_all(data_part, "ParT")

    if data_lgatr is not None:
        print("\n── Fitting LGATr-full probes ───────────────────────────────────")
        results_dict["lgatr"] = fit_all(data_lgatr, "lgatr")
    else:
        results_dict["lgatr"] = None

    if data_slim is not None:
        print("\n── Fitting LGATr-slim probes ───────────────────────────────────")
        results_dict["slim"] = fit_all(data_slim, "slim")
    else:
        results_dict["slim"] = None

    print("\n── Plotting ────────────────────────────────────────────────────")
    plot_curves(results_dict, n_layers_dict, args.output)
    save_scores(results_dict, n_layers_dict, args.output)
    print(f"\nAll results saved to {args.output}")


if __name__ == "__main__":
    main()
