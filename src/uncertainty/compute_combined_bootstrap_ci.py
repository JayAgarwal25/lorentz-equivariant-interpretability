#!/usr/bin/env python3
"""Pool the per-seed bootstrap draws of the three seeds into one distribution and take 95% intervals.

Analyses: probes (h_s probes, all five models, 3 x 200 draws), equivariance (3 x 1000 draws),
grades (grade ablations recomputed from the per-jet logits, 3 x 1000 draws) and mv (L-GATr
invariant probes, 3 x 200 draws). Writes combined_bootstrap_ci.npz (combined_mv_bootstrap_ci.npz
for mv) next to the seed folders, with keys <name>__median, __lo, __hi, __mean, __std, __n_draws.

    python src/uncertainty/compute_combined_bootstrap_ci.py --only probes grades mv
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from uncertainty_utils import bootstrap_indices, percentile_summary

try:
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ── Config ────────────────────────────────────────────────────────────────────

RESULTS = Path(__file__).resolve().parents[2] / "results"
SEEDS = [1001, 1002, 1003]
CI_LO, CI_HI = 2.5, 97.5    # 95% CI
N_BOOTSTRAP_GRADE = 1000     # bootstrap draws per seed for grade recomputation
RNG_SEED = 42


# ── Helpers ───────────────────────────────────────────────────────────────────

def pool_and_ci(arrays: list[np.ndarray]) -> dict:
    """Concatenate arrays along axis 0 and compute CI."""
    pooled = np.concatenate(arrays, axis=0)
    stats = percentile_summary(pooled, low=CI_LO, high=CI_HI)
    stats["n_draws"] = np.array(len(pooled))
    return stats


def save_combined(out_path: Path, results_by_key: dict[str, dict]) -> None:
    flat = {}
    for key, stats in results_by_key.items():
        for stat_name, val in stats.items():
            flat[f"{key}__{stat_name}"] = val
    np.savez_compressed(out_path, **flat)
    print(f"  saved → {out_path.relative_to(RESULTS.parent.parent)}")


# ── 1. Probes (h_s) ───────────────────────────────────────────────────────────
# Raw arrays: (200 draws, 13 layers) per seed per probe target
# Pool 3 seeds → (600, 13), take percentiles → (13,) bounds per layer

def run_probes_hs():
    print("\n=== 1. Probes (h_s) ===")
    models = ["lgatr", "lgatr_slim", "lloca_transformer", "vanilla_transformer", "part"]

    for model in models:
        draws_per_key: dict[str, list] = {}
        for seed in SEEDS:
            path = RESULTS / "probes" / model / f"seed{seed}" / "bootstrap_torch" / "probe_bootstrap_raw.npz"
            f = np.load(path)
            for key in f.files:
                draws_per_key.setdefault(key, []).append(f[key])  # (200, 13)

        out = {key: pool_and_ci(arrays) for key, arrays in draws_per_key.items()}
        save_combined(RESULTS / "probes" / model / "combined_bootstrap_ci.npz", out)


# ── 2. Equivariance ───────────────────────────────────────────────────────────
# Raw arrays: (1000,) logit_errs and (1000, 5) sweep arrays per seed

def _equivariance_paths(model: str) -> list[Path]:
    base = RESULTS / "equivariance" / model
    return [base / f"seed{s}" / "bootstrap" / "equivariance_bootstrap_raw.npz" for s in SEEDS]


def run_equivariance():
    print("\n=== 2. Equivariance ===")
    models = ["lgatr", "lgatr_slim", "lloca_transformer", "vanilla_transformer", "part"]

    for model in models:
        draws_per_key: dict[str, list] = {}
        for path in _equivariance_paths(model):
            f = np.load(path)
            for key in f.files:
                draws_per_key.setdefault(key, []).append(f[key])

        out = {key: pool_and_ci(arrays) for key, arrays in draws_per_key.items()}
        out_dir = RESULTS / "equivariance" / model
        out_dir.mkdir(exist_ok=True)
        save_combined(out_dir / "combined_bootstrap_ci.npz", out)


# ── 3. Grade decomp ───────────────────────────────────────────────────────────
# Raw scores: (10000,) per ablation condition per seed
# Recompute 1000 bootstrap ΔAUC draws per seed, pool 3×1000 → (3000,)
# ΔAUC = AUC(baseline[idx]) − AUC(intervention[idx])

def _bootstrap_grade_deltas(scores_path: Path, intervention_keys: list[str]) -> dict[str, np.ndarray]:
    """Return dict of intervention → (N_BOOTSTRAP_GRADE,) ΔAUC draws for one seed."""
    if not HAS_SKLEARN:
        raise RuntimeError("sklearn required for grade decomp bootstrap; pip install scikit-learn")
    data = np.load(scores_path)
    labels = data["labels"].astype(int)
    baseline = data["baseline"]
    n = len(labels)
    idxs = bootstrap_indices(n, N_BOOTSTRAP_GRADE, RNG_SEED)

    draws: dict[str, np.ndarray] = {k: np.full(N_BOOTSTRAP_GRADE, np.nan) for k in intervention_keys}
    base_draws = np.full(N_BOOTSTRAP_GRADE, np.nan)

    for b, idx in enumerate(idxs):
        y = labels[idx]
        if len(np.unique(y)) < 2:
            continue
        base_auc = roc_auc_score(y, baseline[idx])
        base_draws[b] = base_auc
        for key in intervention_keys:
            if key in data.files:
                auc = roc_auc_score(y, data[key][idx])
                draws[key][b] = base_auc - auc

    draws["baseline_auc"] = base_draws
    return draws


GRADE_CONFIGS = {
    "lgatr": {
        "scores_file": "grade_decomp_scores.npz",
        "interventions": [
            "zero_scalar_like", "zero_vector_like", "zero_bivector",
            "keep_scalar_like", "keep_vector_like", "keep_bivector",
            "zero_scalar_total", "keep_vector_total", "keep_bivector_total",
        ],
    },
    "lgatr_bivector_false": {
        "scores_file": "grade_decomp_scores.npz",
        "interventions": [
            "zero_scalar_like", "zero_vector_like", "zero_bivector",
            "keep_scalar_like", "keep_vector_like", "keep_bivector",
            "zero_scalar_total", "keep_vector_total", "keep_bivector_total",
        ],
    },
    "lgatr_subgroup_false": {
        "scores_file": "grade_decomp_scores.npz",
        "interventions": [
            "zero_g0", "zero_g1", "zero_g2", "zero_g3", "zero_g4",
            "keep_g0", "keep_g1", "keep_g2", "keep_g3", "keep_g4",
        ],
    },
    "lgatr_slim": {
        "scores_file": "grade_decomp_slim_scores.npz",
        "interventions": ["zero_v", "zero_s", "zero_t", "zero_z", "zero_transverse", "zero_spatial"],
    },
}


def run_grade_decomp():
    print("\n=== 3. Grade decomp ===")

    for model, cfg in GRADE_CONFIGS.items():
        print(f"  {model}")
        draws_per_key: dict[str, list] = {}

        for seed in SEEDS:
            scores_path = RESULTS / "ablations" / model / f"seed{seed}" / cfg["scores_file"]
            seed_draws = _bootstrap_grade_deltas(scores_path, cfg["interventions"])
            for key, arr in seed_draws.items():
                draws_per_key.setdefault(key, []).append(arr)  # (1000,) per seed

        out = {key: pool_and_ci(arrays) for key, arrays in draws_per_key.items()}
        save_combined(RESULTS / "ablations" / model / "combined_bootstrap_ci.npz", out)


# ── 4. probe_mv ───────────────────────────────────────────────────────────────
# Raw arrays: (200 draws, 13 layers) per seed, 88 keys (8 targets × 11 grade groups)
# lgatr only — only model with multivector hidden state
# Pool 3 seeds → (600, 13), take percentiles → (13,) bounds per layer

def run_probe_mv():
    print("\n=== 4. probe_mv (lgatr only) ===")
    draws_per_key: dict[str, list] = {}

    for seed in SEEDS:
        path = RESULTS / "probes" / "lgatr" / f"seed{seed}" / "mv_bootstrap" / "invariant_probe_bootstrap_raw.npz"
        f = np.load(path)
        for key in f.files:
            draws_per_key.setdefault(key, []).append(f[key])  # (200, 13)

    out = {key: pool_and_ci(arrays) for key, arrays in draws_per_key.items()}
    save_combined(RESULTS / "probes" / "lgatr" / "combined_mv_bootstrap_ci.npz", out)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=["probes", "equivariance", "grades", "mv"],
                    default=["probes", "equivariance", "grades", "mv"])
    only = ap.parse_args().only

    print("Computing combined bootstrap CIs")
    print(f"  CI level: {CI_LO}th–{CI_HI}th percentile (95%)")
    print(f"  Seeds: {SEEDS}")
    print(f"  Grade bootstrap draws per seed: {N_BOOTSTRAP_GRADE}")

    if "probes" in only:
        run_probes_hs()
    if "equivariance" in only:
        run_equivariance()
    if "grades" in only:
        if HAS_SKLEARN:
            run_grade_decomp()
        else:
            print("\n=== 3. Grade decomp — SKIPPED (sklearn not available) ===")
            print("  Install with: pip install scikit-learn")
    if "mv" in only:
        run_probe_mv()

    print("\nDone.")
    print("CI format for tables: median_{-diff_lo}^{+diff_hi}")
    print("  diff_lo = median - lo  (downward uncertainty)")
    print("  diff_hi = hi - median  (upward uncertainty)")
