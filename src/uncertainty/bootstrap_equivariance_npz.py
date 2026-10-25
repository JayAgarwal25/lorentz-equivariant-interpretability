#!/usr/bin/env python3
"""Bootstrap uncertainty for saved equivariance-error NPZ files.

This is model-agnostic: any numeric array with shape
  (n_jets, n_transforms, ...)
or
  (n_jets, ...)
can be bootstrapped by resampling jets on axis 0.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from uncertainty_utils import bootstrap_indices, bootstrap_mean, percentile_summary


def _is_event_array(arr):
    return np.issubdtype(arr.dtype, np.number) and arr.ndim >= 2 and arr.shape[0] > 1


def _bootstrap_equivariance_array(key, arr, n_bootstrap, seed):
    """Return bootstrap samples for common equivariance array layouts."""
    arr = arr.astype(np.float64)
    if key.startswith("sweep") and arr.ndim == 3:
        # Sweep arrays are shaped (gamma, jets, transforms). Resample jets.
        idxs = bootstrap_indices(arr.shape[1], n_bootstrap, seed)
        return np.stack([np.nanmean(arr[:, idx, :], axis=(1, 2)) for idx in idxs], axis=0)
    if arr.ndim == 2:
        # Standard logit errors: (jets, transforms) -> scalar mean.
        return bootstrap_mean(arr, n_bootstrap=n_bootstrap, seed=seed, reduce_axes=(0, 1))
    if arr.ndim >= 3:
        # Standard hidden errors: (jets, transforms, layers...) -> per-layer mean.
        return bootstrap_mean(arr, n_bootstrap=n_bootstrap, seed=seed, reduce_axes=(0, 1))
    raise ValueError(f"Unsupported array layout for {key}: {arr.shape}")


def bootstrap_file(path, n_bootstrap, seed, low, high):
    data = np.load(path, allow_pickle=False)
    raw = {}
    summary = {}

    for key in data.files:
        arr = data[key]
        if not _is_event_array(arr):
            continue
        samples = _bootstrap_equivariance_array(key, arr, n_bootstrap, seed)
        raw[key] = samples
        summary[key] = percentile_summary(samples, low=low, high=high)

    return raw, summary


def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Equivariance NPZ with per-jet errors.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--low", type=float, default=16.0)
    ap.add_argument("--high", type=float, default=84.0)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw, summary = bootstrap_file(args.input, args.n_bootstrap, args.seed, args.low, args.high)
    np.savez_compressed(out / "equivariance_bootstrap_raw.npz", **raw)

    flat = {}
    for key, stats in summary.items():
        for stat_name, values in stats.items():
            flat[f"{key}__{stat_name}"] = values
    np.savez_compressed(out / "equivariance_bootstrap_summary.npz", **flat)

    meta = {
        "input": str(args.input),
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "interval_percentiles": [args.low, args.high],
        "keys": sorted(summary),
    }
    (out / "equivariance_bootstrap_summary.json").write_text(
        json.dumps({"meta": meta, "summary": _jsonable(summary)}, indent=2)
    )
    print(f"Wrote bootstrap equivariance uncertainty to {out}")


if __name__ == "__main__":
    main()
