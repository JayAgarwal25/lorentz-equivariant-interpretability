#!/usr/bin/env python3
"""GPU bootstrap for L-GATr MV invariant probes.

Mirrors bootstrap_invariant_probe_npz.py but uses torch/GPU for Ridge and LBFGS
logistic regression instead of sklearn. Same output format.

Input NPZs:
  --invariants   probe_data_lgatr_mv_invariants.npz   (jet_inv: L x N x 832)
  --targets      probe_data_lgatr.npz                 (jet labels/targets)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score, roc_auc_score

from uncertainty_utils import bootstrap_indices, percentile_summary


TARGETS = [
    ("top_qcd",   "binary_cls", "jet_labels"),
    ("jet_mass",  "regression", "jet_mass"),
    ("jet_mult",  "regression", "jet_multiplicity"),
    ("jet_tau1",  "regression", "jet_tau1"),
    ("jet_tau2",  "regression", "jet_tau2"),
    ("jet_tau3",  "regression", "jet_tau3"),
    ("jet_tau21", "regression", "jet_tau21"),
    ("jet_tau32", "regression", "jet_tau32"),
]

GROUPS = {
    "scalar_like":   (0,   32),
    "vector_like":   (32,  560),
    "bivector_like": (560, 832),
    "All":           (0,   832),
    "G0":            (0,   16),
    "G4":            (16,  32),
    "G1":            (32,  168),
    "G3":            (168, 304),
    "Cross":         (304, 560),
    "G2_scalar":     (560, 696),
    "G2_pseudo":     (696, 832),
}


def _standardize(X_tr, X_te):
    mu  = X_tr.mean(0)
    sig = X_tr.std(0).clamp(min=1e-8)
    constant = (X_tr.max(0).values == X_tr.min(0).values)
    Xs  = (X_tr - mu) / sig;  Xs[:, constant]  = 0.0
    Xts = (X_te - mu) / sig;  Xts[:, constant] = 0.0
    return Xs, Xts


def ridge_predict(X_tr, y_tr, X_te, alpha):
    Xs, Xts = _standardize(X_tr, X_te)
    y_mean = y_tr.mean()
    yc = y_tr - y_mean
    A  = Xs.T @ Xs
    A.diagonal().add_(alpha)
    w  = torch.linalg.solve(A, (Xs.T @ yc).unsqueeze(1)).squeeze(1)
    return (Xts @ w + y_mean).detach().cpu().numpy()


def logreg_predict_proba(X_tr, y_tr, X_te, c_value, max_iter):
    Xs, Xts = _standardize(X_tr, X_te)
    n_tr, dim = Xs.shape
    W = torch.zeros(2, dim, device=Xs.device, requires_grad=True)
    b = torch.zeros(2,      device=Xs.device, requires_grad=True)
    opt = torch.optim.LBFGS(
        [W, b], lr=1.0, max_iter=max_iter,
        tolerance_grad=1e-6, tolerance_change=1e-9,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(Xs @ W.T + b, y_tr) + W.pow(2).sum() / (2.0 * c_value * n_tr)
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return F.softmax(Xts @ W.T + b, dim=1).cpu().numpy()


def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--invariants",    required=True)
    ap.add_argument("--targets",       required=True)
    ap.add_argument("--output-dir",    required=True)
    ap.add_argument("--n-bootstrap",   type=int,   default=200)
    ap.add_argument("--seed",          type=int,   default=42)
    ap.add_argument("--only-targets",  nargs="+",  default=None)
    ap.add_argument("--only-groups",   nargs="+",  default=None)
    ap.add_argument("--device",        default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ridge-alpha",   type=float, default=10.0)
    ap.add_argument("--logreg-c",      type=float, default=0.1)
    ap.add_argument("--lbfgs-max-iter",type=int,   default=100)
    ap.add_argument("--report-every",  type=int,   default=10)
    ap.add_argument("--low",           type=float, default=16.0)
    ap.add_argument("--high",          type=float, default=84.0)
    args = ap.parse_args()

    device = torch.device(args.device)

    inv_npz  = np.load(args.invariants, allow_pickle=False)
    jet_inv  = inv_npz["jet_inv"].astype(np.float32)          # (L, N, 832)
    n_layers, n_jets, _ = jet_inv.shape

    tgt_npz = dict(np.load(args.targets, allow_pickle=False))
    if "jet_tau32" not in tgt_npz and "jet_tau3" in tgt_npz and "jet_tau2" in tgt_npz:
        tgt_npz["jet_tau32"] = tgt_npz["jet_tau3"] / (tgt_npz["jet_tau2"] + 1e-10)
    if "jet_tau21" not in tgt_npz and "jet_tau1" in tgt_npz and "jet_tau2" in tgt_npz:
        tgt_npz["jet_tau21"] = tgt_npz["jet_tau2"] / (tgt_npz["jet_tau1"] + 1e-10)

    targets = [t for t in TARGETS if args.only_targets is None or t[0] in args.only_targets]
    groups  = {k: v for k, v in GROUPS.items() if args.only_groups  is None or k in args.only_groups}

    idxs = bootstrap_indices(n_jets, args.n_bootstrap, args.seed)
    samples = {
        f"{tn}__{gn}": np.full((args.n_bootstrap, n_layers), np.nan)
        for tn, *_ in targets
        for gn in groups
    }

    print(f"Device: {device}")
    print(f"jet_inv: {jet_inv.shape}  bootstraps={args.n_bootstrap}  "
          f"targets={len(targets)}  groups={len(groups)}  layers={n_layers}")

    for b, boot_jets in enumerate(idxs):
        oob_mask = np.ones(n_jets, dtype=bool)
        oob_mask[np.unique(boot_jets)] = False
        oob = np.where(oob_mask)[0]
        if len(oob) == 0:
            continue

        for tn, kind, key in targets:
            y_all = tgt_npz[key]
            y_tr_np = y_all[boot_jets]
            y_te    = y_all[oob]
            if kind != "regression" and (
                len(np.unique(y_tr_np)) < 2 or len(np.unique(y_te)) < 2
            ):
                continue

            if kind == "regression":
                y_tr = torch.as_tensor(y_tr_np.astype(np.float32), device=device)
            else:
                y_tr = torch.as_tensor(y_tr_np.astype(np.int64), device=device)

            for gn, (s, e) in groups.items():
                out_key = f"{tn}__{gn}"
                inv_slice = jet_inv[:, :, s:e]  # (L, N, feat)
                for li in range(n_layers):
                    X_tr = torch.as_tensor(inv_slice[li, boot_jets], device=device)
                    X_te = torch.as_tensor(inv_slice[li, oob],       device=device)
                    if kind == "regression":
                        pred = ridge_predict(X_tr, y_tr, X_te, args.ridge_alpha)
                        samples[out_key][b, li] = r2_score(y_te, pred)
                    else:
                        proba = logreg_predict_proba(X_tr, y_tr, X_te,
                                                     args.logreg_c, args.lbfgs_max_iter)
                        samples[out_key][b, li] = roc_auc_score(y_te, proba[:, 1])

        if (b + 1) % args.report_every == 0:
            print(f"  {b + 1}/{args.n_bootstrap}")

    summary = {k: percentile_summary(v, args.low, args.high) for k, v in samples.items()}
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "invariant_probe_bootstrap_raw.npz",     **samples)
    flat = {f"{k}__{stat}": val for k, stats in summary.items() for stat, val in stats.items()}
    np.savez_compressed(out / "invariant_probe_bootstrap_summary.npz", **flat)
    (out / "invariant_probe_bootstrap_summary.json").write_text(
        json.dumps(
            {
                "meta": {
                    "invariants":          args.invariants,
                    "targets":             args.targets,
                    "n_bootstrap":         args.n_bootstrap,
                    "seed":                args.seed,
                    "device":              args.device,
                    "interval_percentiles":[args.low, args.high],
                },
                "summary": _jsonable(summary),
            },
            indent=2,
        )
    )
    print(f"Wrote invariant probe bootstrap to {out}")


if __name__ == "__main__":
    main()
