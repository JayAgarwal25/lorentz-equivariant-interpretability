#!/usr/bin/env python3
"""Fit linear probes on the saved representations of Vanilla-T or LLoCa-T, per layer and target.

    python src/transformer/fit_probes_vanilla.py --train-jets 8000 \\
        --data results/probes/vanilla_transformer/seed1001/probe_data_vanilla_transformer.npz \\
        --output results/probes/vanilla_transformer/seed1001/probe_results
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, r2_score
import json

TARGETS = [
    ("top_qcd",        "binary_cls",  "jet", "jet_labels", "top/QCD"),
    ("jet_mass",       "regression",  "jet", "jet_mass", "Jet Mass"),
    ("jet_mult",       "regression",  "jet", "jet_multiplicity", "Multiplicity"),
    ("jet_tau1",       "regression",  "jet", "jet_tau1", r"$\tau_1$"),
    ("jet_tau2",       "regression",  "jet", "jet_tau2", r"$\tau_2$"),
    ("jet_tau3",       "regression",  "jet", "jet_tau3", r"$\tau_3$"),
    ("jet_tau21",      "regression",  "jet", "jet_tau21", r"$\tau_{21}$"),
    ("jet_tau32",      "regression",  "jet", "jet_tau32", r"$\tau_{32}$"),
    ("part_pt",        "regression",  "particle", "particle_pt", r"$p_T$"),
    ("part_eta",       "regression",  "particle", "particle_eta", r"$\eta$"),
    ("part_phi",       "regression",  "particle", "particle_phi", r"$\phi$"),
    ("part_E",         "regression",  "particle", "particle_E", r"$E$"),
    ("part_quartile",  "multi_cls",   "particle", "particle_pt_rank_quartile", r"$p_T$ Quartile"),
    ("part_dR",        "regression",  "particle", "particle_dR_to_axis", r"$\Delta R$"),
]

MAX_PARTICLES_FIT = 150_000


def load_probe_data(path):
    npz = np.load(path)
    data = {k: npz[k] for k in npz.files}
    if "jet_tau21" not in data and {"jet_tau1", "jet_tau2"} <= data.keys():
        data["jet_tau21"] = data["jet_tau2"] / (data["jet_tau1"] + 1e-10)
    if "jet_tau32" not in data and {"jet_tau2", "jet_tau3"} <= data.keys():
        data["jet_tau32"] = data["jet_tau3"] / (data["jet_tau2"] + 1e-10)
    return data

def particle_split(jet_train_idx, jet_test_idx, part_jet_idx):
    train_set = set(jet_train_idx.tolist())
    test_set = set(jet_test_idx.tolist())
    all_idx = np.arange(len(part_jet_idx))
    tr = all_idx[np.isin(part_jet_idx, list(train_set))]
    te = all_idx[np.isin(part_jet_idx, list(test_set))]
    return tr, te

def subsample(idx, max_n, seed=0):
    if len(idx) <= max_n:
        return idx
    rng = np.random.default_rng(seed)
    return rng.choice(idx, size=max_n, replace=False)

def make_probe(kind):
    if kind == "regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ])
    else:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, solver='lbfgs', C=1.0, n_jobs=-1)),
        ])

def score_probe(kind, y_true, y_pred_or_proba):
    if kind == "regression":
        return r2_score(y_true, y_pred_or_proba)
    elif kind == "binary_cls":
        return roc_auc_score(y_true, y_pred_or_proba[:, 1])
    else:  # multi_cls
        return accuracy_score(y_true, y_pred_or_proba)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="probe_data_vanilla_topt.npz")
    ap.add_argument("--output", default="results/probe_results_vanilla/")
    ap.add_argument("--show-dims", action="store_true", help="Print data dimensions and exit")
    ap.add_argument("--only", nargs="+", default=None, help="Run only these probe names")
    ap.add_argument(
        "--train-jets",
        type=int,
        default=8000,
        help="Number of leading jets to use for training; the remainder are held out for testing.",
    )
    ap.add_argument(
        "--max-particles-fit",
        type=int,
        default=MAX_PARTICLES_FIT,
        help="Maximum number of training particles to use for particle-level probes.",
    )
    args = ap.parse_args()

    data = load_probe_data(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded data from {args.data}")
    print(f"Keys: {list(data.keys())}")

    if args.show_dims:
        print("\nData dimensions:")
        for k in data.keys():
            print(f"  {k:30s}: {str(data[k].shape):20s} {data[k].dtype}")
        return

    # Filter out jets with NaN in representations
    jet_reps = data['jet_reps']  # (11, n_jets, 128)
    nan_mask = np.isnan(jet_reps).any(axis=(0, 2))  # NaN in any layer or feature
    good_jets = np.where(~nan_mask)[0]
    print(f"Filtering out jets with NaN: {len(nan_mask) - len(good_jets)}/{len(nan_mask)} removed")

    n_jets = len(data['jet_labels'])
    if args.train_jets <= 0 or args.train_jets >= n_jets:
        raise ValueError(
            f"--train-jets must be in [1, {n_jets - 1}] for this file; got {args.train_jets}"
        )

    # The first --train-jets jets are the training set, the rest the test set.
    n_layers = data['jet_reps'].shape[0]
    train_idx_all = np.arange(args.train_jets)
    test_idx_all = np.arange(args.train_jets, n_jets)

    # Keep only good jets
    train_idx = np.intersect1d(train_idx_all, good_jets)
    test_idx = np.intersect1d(test_idx_all, good_jets)
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            f"NaN filtering left an empty split: train={len(train_idx)}, test={len(test_idx)}"
        )
    print(
        f"Train jets: {len(train_idx)}/{len(train_idx_all)}, "
        f"Test jets: {len(test_idx)}/{len(test_idx_all)}"
    )

    part_jet_idx = data['particle_jet_idx']
    ptr, pte = particle_split(train_idx, test_idx, part_jet_idx)
    ptr_sub = subsample(ptr, args.max_particles_fit)

    results = {}

    active_targets = [t for t in TARGETS if args.only is None or t[0] in args.only]
    for probe_name, probe_type, level, label_key, display_name in active_targets:
        print(f"\n{probe_name}...")
        scores = np.full(n_layers, np.nan)
        y_all = data[label_key]

        for li in range(n_layers):
            if level == "jet":
                X_all = data['jet_reps'][li]  # (n_jets, 128)
                X_train = X_all[train_idx]
                X_test = X_all[test_idx]
                y_train = y_all[train_idx]
                y_test = y_all[test_idx]
            else:
                X_all = data['particle_reps'][li]  # (n_particles, 128)
                X_train = X_all[ptr_sub]
                X_test = X_all[pte]
                y_train = y_all[ptr_sub]
                y_test = y_all[pte]

            probe = make_probe(probe_type)
            probe.fit(X_train, y_train)

            if probe_type == "regression":
                y_pred = probe.predict(X_test)
                score = score_probe(probe_type, y_test, y_pred)
            elif probe_type == "binary_cls":
                y_proba = probe.predict_proba(X_test)
                score = score_probe(probe_type, y_test, y_proba)
            else:  # multi_cls
                y_pred = probe.predict(X_test)
                score = score_probe(probe_type, y_test, y_pred)

            scores[li] = score

        results[probe_name] = scores
        layer_str = "  ".join([f"L{i}={s:.3f}" for i, s in enumerate(scores)])
        print(f"  {layer_str}")

    # Save results
    with open(output_dir / "probe_scores_vanilla.txt", "w") as f:
        f.write("Per-layer probe scores for vanilla transformer (TopTagging test set)\n")
        f.write("="*80 + "\n\n")
        for probe_name, probe_type, level, label_key, display_name in active_targets:
            scores = results[probe_name]
            f.write(f"{display_name:40s}: " + "  ".join([f"L{i}={s:.3f}" for i, s in enumerate(scores)]) + "\n")

    with open(output_dir / "probe_scores_vanilla.json", "w") as f:
        json.dump({k: v.tolist() for k, v in results.items()}, f, indent=2)

    np.savez(output_dir / "probe_scores.npz", **results)

    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    main()
