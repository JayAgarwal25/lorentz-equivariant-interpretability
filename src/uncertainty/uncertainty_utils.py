#!/usr/bin/env python3
"""Small shared helpers for bootstrap and seed-band uncertainty.

The workshop scripts use two different uncertainty notions:
  1. bootstrap over evaluated jets/events for one fixed checkpoint;
  2. seed aggregation over independently trained checkpoints.
Keeping these helpers tiny makes that distinction explicit in every caller.
"""

from __future__ import annotations

import numpy as np


def percentile_summary(samples: np.ndarray, low: float = 16.0, high: float = 84.0) -> dict:
    """Return median and central percentile interval along bootstrap/seed axis 0."""
    return {
        "median": np.nanmedian(samples, axis=0),
        "lo": np.nanpercentile(samples, low, axis=0),
        "hi": np.nanpercentile(samples, high, axis=0),
        "mean": np.nanmean(samples, axis=0),
        "std": np.nanstd(samples, axis=0),
    }


def bootstrap_indices(n_items: int, n_bootstrap: int, seed: int) -> np.ndarray:
    """Draw bootstrap index arrays with replacement."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_items, size=(n_bootstrap, n_items), dtype=np.int64)


def bootstrap_mean(
    values: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
    reduce_axes: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Bootstrap a mean after resampling axis 0.

    Parameters
    ----------
    values:
        Array whose first axis is the independently resampled event/jet axis.
    reduce_axes:
        Axes to average after resampling. If omitted, all axes except the last
        are averaged, which is convenient for arrays shaped
        (jets, transforms, layers).
    """
    arr = np.asarray(values)
    if reduce_axes is None:
        reduce_axes = tuple(range(arr.ndim - 1))

    idxs = bootstrap_indices(arr.shape[0], n_bootstrap, seed)
    out = []
    for idx in idxs:
        out.append(np.nanmean(arr[idx], axis=reduce_axes))
    return np.stack(out, axis=0)

