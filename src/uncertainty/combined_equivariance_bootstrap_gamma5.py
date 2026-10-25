#!/usr/bin/env python3
"""Combined bootstrap CI for equivariance_gamma5 results.

Pools all jets across 3 seeds per model, bootstraps on GPU, outputs 95% CI.
No intermediate per-seed bootstrap step required.

Array conventions (per model output NPZ):
  Regular  : (jets, transforms [, layers]) — jets on axis 0
  Sweep    : (gammas, jets_sweep, transforms) — jets on axis 1
  Metadata : (gammas,) 1-D — saved as-is, no bootstrap

Output: equivariance_gamma5/{model}/combined_bootstrap_ci.npz
  Keys: {orig_key}__median, __lo, __hi, __mean, __std, __n_draws
        gammas  (plain array, no suffix)
"""

from pathlib import Path

import numpy as np
import torch

RESULTS = Path(__file__).resolve().parents[2] / "results"
EQ_DIR  = RESULTS / "equivariance_gamma5"
SEEDS   = [1001, 1002, 1003]

N_BOOTSTRAP = 1000
CI_LO, CI_HI = 2.5, 97.5
RNG_SEED = 42

NPZ_NAMES = {
    "lgatr":                        "equivariance_errors.npz",
    "lgatr_slim":                   "equivariance_slim_errors.npz",
    "vanilla_transformer":          "equivariance_vanilla_topt.npz",
    "lloca_transformer_full_event": "equivariance_lloca_topt.npz",
    "part":                         "equivariance_part.npz",
}


def _is_metadata(arr: np.ndarray) -> bool:
    return arr.ndim <= 1


def _is_sweep(key: str, arr: np.ndarray) -> bool:
    return key.startswith("sweep") and arr.ndim == 3


def pool_seeds(model: str, key: str) -> np.ndarray:
    arrays = []
    for seed in SEEDS:
        path = EQ_DIR / model / f"seed{seed}" / NPZ_NAMES[model]
        arrays.append(np.load(path)[key])
    # sweep (gammas, jets, transforms) → concat on jets axis (1)
    # all others (jets, ...) → concat on jets axis (0)
    axis = 1 if _is_sweep(key, arrays[0]) else 0
    return np.concatenate(arrays, axis=axis)


def bootstrap_gpu(pooled: np.ndarray, key: str, device: torch.device,
                  rng: np.random.Generator) -> np.ndarray:
    """Return (N_BOOTSTRAP, ...) bootstrap mean draws on GPU."""
    if _is_sweep(key, pooled):
        # pooled: (G, J, T)  →  draws: (B, G)
        G, J, T = pooled.shape
        t = torch.from_numpy(pooled.astype(np.float32)).to(device)
        idxs = torch.from_numpy(rng.integers(0, J, size=(N_BOOTSTRAP, J))).to(device)
        # t[:, idxs, :] → (G, B, J, T)
        sampled = t[:, idxs, :]
        draws = sampled.mean(dim=(2, 3)).T  # (B, G)
    elif pooled.ndim == 2:
        # (J, T)  →  draws: (B,)  [scalar per draw]
        J = pooled.shape[0]
        t = torch.from_numpy(pooled.astype(np.float32)).to(device)
        idxs = torch.from_numpy(rng.integers(0, J, size=(N_BOOTSTRAP, J))).to(device)
        sampled = t[idxs]           # (B, J, T)
        draws = sampled.mean(dim=(1, 2))  # (B,)
    else:
        # (J, T, L)  →  draws: (B, L)  [per-layer mean]
        J = pooled.shape[0]
        t = torch.from_numpy(pooled.astype(np.float32)).to(device)
        idxs = torch.from_numpy(rng.integers(0, J, size=(N_BOOTSTRAP, J))).to(device)
        sampled = t[idxs]           # (B, J, T, L)
        draws = sampled.mean(dim=(1, 2))  # (B, L)

    return draws.cpu().numpy()


def percentile_ci(draws: np.ndarray) -> dict:
    return {
        "median":  np.nanpercentile(draws, 50,    axis=0),
        "lo":      np.nanpercentile(draws, CI_LO,  axis=0),
        "hi":      np.nanpercentile(draws, CI_HI,  axis=0),
        "mean":    np.nanmean(draws, axis=0),
        "std":     np.nanstd(draws,  axis=0),
        "n_draws": np.array(len(draws)),
    }


def run_model(model: str, device: torch.device, rng: np.random.Generator) -> None:
    print(f"\n[{model}]")
    ref = np.load(EQ_DIR / model / f"seed{SEEDS[0]}" / NPZ_NAMES[model])

    flat = {}
    for key in ref.files:
        arr0 = ref[key]
        if _is_metadata(arr0):
            flat[key] = arr0
            print(f"  {key}: metadata {arr0.shape} — saved as-is")
            continue

        pooled = pool_seeds(model, key)
        draws  = bootstrap_gpu(pooled, key, device, rng)
        ci     = percentile_ci(draws)

        print(f"  {key}: pooled {pooled.shape} → draws {draws.shape} "
              f"median={ci['median'].mean():.3e}")
        for stat, val in ci.items():
            flat[f"{key}__{stat}"] = val

    out_path = EQ_DIR / model / "combined_bootstrap_ci.npz"
    np.savez_compressed(out_path, **flat)
    print(f"  saved → {out_path}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  N_BOOTSTRAP={N_BOOTSTRAP}  CI=[{CI_LO},{CI_HI}]%")
    rng = np.random.default_rng(RNG_SEED)
    for model in NPZ_NAMES:
        run_model(model, device, rng)
    print("\nAll done.")


if __name__ == "__main__":
    main()
