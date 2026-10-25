#!/usr/bin/env python3
"""Bootstrap the linear probes of one probe-data file on the GPU (Ridge and logistic regression in torch).

Each draw fits every probe at every layer on jets resampled with replacement and scores it on the
out-of-bag jets.

    python src/uncertainty/bootstrap_probe_npz_torch.py --input results/probes/lgatr/seed1001/probe_data_lgatr.npz \\
        --output-dir results/probes/lgatr/seed1001/bootstrap_torch --n-bootstrap 200 --seed 1001 \\
        --max-train-particles 150000 --max-test-particles 150000 --lbfgs-max-iter 100
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score

from uncertainty_utils import bootstrap_indices, percentile_summary


TARGETS = [
    ("top_qcd", "binary_cls", "jet", "jet_labels"),
    ("jet_mass", "regression", "jet", "jet_mass"),
    ("jet_mult", "regression", "jet", "jet_multiplicity"),
    ("jet_tau1", "regression", "jet", "jet_tau1"),
    ("jet_tau2", "regression", "jet", "jet_tau2"),
    ("jet_tau3", "regression", "jet", "jet_tau3"),
    ("jet_tau21", "regression", "jet", "jet_tau21"),
    ("jet_tau32", "regression", "jet", "jet_tau32"),
    ("part_pt", "regression", "particle", "particle_pt"),
    ("part_eta", "regression", "particle", "particle_eta"),
    ("part_phi", "regression", "particle", "particle_phi"),
    ("part_E", "regression", "particle", "particle_E"),
    ("part_quartile", "multi_cls", "particle", "particle_pt_rank_quartile"),
    ("part_dR", "regression", "particle", "particle_dR_to_axis"),
]


def _augment_targets(data):
    if "jet_tau21" not in data and {"jet_tau1", "jet_tau2"} <= data.keys():
        data["jet_tau21"] = data["jet_tau2"] / (data["jet_tau1"] + 1e-10)
    if "jet_tau32" not in data and {"jet_tau2", "jet_tau3"} <= data.keys():
        data["jet_tau32"] = data["jet_tau3"] / (data["jet_tau2"] + 1e-10)
    return data


def _standardize(X_tr, X_te):
    mu = X_tr.mean(0)
    sig = X_tr.std(0).clamp(1e-8)
    constant = X_tr.max(0).values == X_tr.min(0).values
    Xs = (X_tr - mu) / sig
    Xts = (X_te - mu) / sig
    Xs[:, constant] = 0.0
    Xts[:, constant] = 0.0
    return Xs, Xts


def ridge_predict(X_tr, y_tr, X_te, alpha):
    Xs, Xts = _standardize(X_tr, X_te)
    y_mean = y_tr.mean()
    yc = y_tr - y_mean
    A = Xs.T @ Xs
    A.diagonal().add_(alpha)
    b = Xs.T @ yc
    w = torch.linalg.solve(A, b.unsqueeze(1)).squeeze(1)
    return (Xts @ w + y_mean).detach().cpu().numpy()


def logreg_predict_proba(X_tr, y_tr, X_te, n_classes, c_value, max_iter):
    Xs, Xts = _standardize(X_tr, X_te)
    n_tr, dim = Xs.shape
    W = torch.zeros(n_classes, dim, device=Xs.device, requires_grad=True)
    b = torch.zeros(n_classes, device=Xs.device, requires_grad=True)
    opt = torch.optim.LBFGS(
        [W, b],
        lr=1.0,
        max_iter=max_iter,
        tolerance_grad=1e-6,
        tolerance_change=1e-9,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad()
        logits = Xs @ W.T + b
        loss = F.cross_entropy(logits, y_tr) + W.pow(2).sum() / (2.0 * c_value * n_tr)
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return F.softmax(Xts @ W.T + b, dim=1).cpu().numpy()


def jet_particle_buckets(particle_jet_idx, n_jets):
    buckets = [[] for _ in range(n_jets)]
    for pidx, jidx in enumerate(particle_jet_idx.astype(int)):
        if 0 <= jidx < n_jets:
            buckets[jidx].append(pidx)
    return [np.asarray(x, dtype=np.int64) for x in buckets]


def concat_particles(buckets, jets, max_particles, rng):
    arrays = [buckets[j] for j in jets if len(buckets[j])]
    if not arrays:
        return np.asarray([], dtype=np.int64)
    idx = np.concatenate(arrays)
    if max_particles and len(idx) > max_particles:
        idx = rng.choice(idx, size=max_particles, replace=False)
    return idx


def fit_bootstrap(data, args, boot_jets, oob_jets, buckets, targets, seed_b):
    rng = np.random.default_rng(seed_b)
    n_layers = data["jet_reps"].shape[0]
    tr_parts = concat_particles(buckets, boot_jets, args.max_train_particles, rng)
    te_parts = concat_particles(buckets, oob_jets, args.max_test_particles, rng)
    result = {name: np.full(n_layers, np.nan) for name, *_ in targets}

    for name, kind, level, label_key in targets:
        y_all = data[label_key]
        if level == "jet":
            tr, te = boot_jets, oob_jets
            X_source = data["jet_reps"]
        else:
            tr, te = tr_parts, te_parts
            X_source = data["particle_reps"]
        if len(tr) == 0 or len(te) == 0:
            continue
        y_tr_np = y_all[tr]
        y_te = y_all[te]
        if kind != "regression" and (len(np.unique(y_tr_np)) < 2 or len(np.unique(y_te)) < 2):
            continue

        y_tr = torch.as_tensor(
            y_tr_np.astype(np.float32 if kind == "regression" else np.int64),
            device=args.device,
        )
        n_classes = int(np.nanmax(y_all)) + 1 if kind == "multi_cls" else 2

        for li in range(n_layers):
            X_tr = torch.as_tensor(X_source[li, tr].astype(np.float32), device=args.device)
            X_te = torch.as_tensor(X_source[li, te].astype(np.float32), device=args.device)
            if kind == "regression":
                pred = ridge_predict(X_tr, y_tr, X_te, args.ridge_alpha)
                result[name][li] = r2_score(y_te, pred)
            elif kind == "binary_cls":
                proba = logreg_predict_proba(X_tr, y_tr, X_te, 2, args.logreg_c, args.lbfgs_max_iter)
                result[name][li] = roc_auc_score(y_te, proba[:, 1])
            else:
                proba = logreg_predict_proba(
                    X_tr, y_tr, X_te, n_classes, args.logreg_c, args.lbfgs_max_iter
                )
                result[name][li] = accuracy_score(y_te, proba.argmax(axis=1))
    return result


def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--n-bootstrap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-train-particles", type=int, default=150_000)
    ap.add_argument("--max-test-particles", type=int, default=150_000)
    ap.add_argument("--ridge-alpha", type=float, default=1.0)
    ap.add_argument("--logreg-c", type=float, default=1.0)
    ap.add_argument("--lbfgs-max-iter", type=int, default=100)
    ap.add_argument("--report-every", type=int, default=10)
    ap.add_argument("--low", type=float, default=16.0)
    ap.add_argument("--high", type=float, default=84.0)
    args = ap.parse_args()

    npz = np.load(args.input, allow_pickle=False)
    data = _augment_targets({k: npz[k] for k in npz.files})
    n_jets = data["jet_reps"].shape[1]
    n_layers = data["jet_reps"].shape[0]
    targets = [t for t in TARGETS if args.only is None or t[0] in args.only]
    buckets = jet_particle_buckets(data["particle_jet_idx"], n_jets)
    idxs = bootstrap_indices(n_jets, args.n_bootstrap, args.seed)
    samples = {name: np.full((args.n_bootstrap, n_layers), np.nan) for name, *_ in targets}

    print(f"Input: {args.input}")
    print(f"Device: {args.device}; bootstraps={args.n_bootstrap}; layers={n_layers}")
    for b, boot_jets in enumerate(idxs):
        oob_mask = np.ones(n_jets, dtype=bool)
        oob_mask[np.unique(boot_jets)] = False
        oob_jets = np.where(oob_mask)[0]
        if len(oob_jets) == 0:
            continue
        result = fit_bootstrap(data, args, boot_jets, oob_jets, buckets, targets, args.seed + b)
        for name, values in result.items():
            samples[name][b] = values
        if (b + 1) % args.report_every == 0:
            print(f"  {b + 1}/{args.n_bootstrap}")

    summary = {name: percentile_summary(values, args.low, args.high) for name, values in samples.items()}
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "probe_bootstrap_raw.npz", **samples)
    flat = {
        f"{name}__{stat}": value
        for name, stats in summary.items()
        for stat, value in stats.items()
    }
    np.savez_compressed(out / "probe_bootstrap_summary.npz", **flat)
    (out / "probe_bootstrap_summary.json").write_text(
        json.dumps(
            {
                "meta": {
                    "input": args.input,
                    "n_bootstrap": args.n_bootstrap,
                    "seed": args.seed,
                    "device": args.device,
                    "interval_percentiles": [args.low, args.high],
                },
                "summary": _jsonable(summary),
            },
            indent=2,
        )
    )
    print(f"Wrote Torch probe bootstrap uncertainty to {out}")


if __name__ == "__main__":
    main()
