#!/usr/bin/env python
"""Per-seed bootstrap confidence intervals and one-sided p-values for grade-ablation AUC drops.

Reads the per-jet logits saved by run_grade_decomp_lgatr.py or run_grade_decomp_slim.py with
--save-scores. Modes: workshop (grouped L-GATr ablations), all_grades (five separate grades),
slim (L-GATr-slim channel ablations). The intervals in the paper pool the three seeds with
src/uncertainty/compute_combined_bootstrap_ci.py.

    python src/lgatr/bootstrap_grade_significance.py --scores results/ablations/lgatr/seed1001/grade_decomp_scores.npz \\
        --mode workshop --label L-GATr --n-bootstrap 1000 --seed 1001 --output results/ablations/lgatr/seed1001
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score

# ── Grade / ablation metadata ─────────────────────────────────────────────────

GRADE_NAMES = ["G0 scalar", "G1 vector", "G2 bivector", "G3 trivector", "G4 pseudoscalar"]
GRADE_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

SLIM_ABL_NAMES  = ["zero_v", "zero_s", "zero_t", "zero_z", "zero_transverse", "zero_spatial"]
SLIM_ABL_LABELS = ["Zero h_v\n(vectors)", "Zero h_s\n(scalars)", "Zero e_t\n(energy)",
                   "Zero e_z\n(beam)", "Zero e_x,e_y\n(transverse)", "Zero e_x,e_y,e_z\n(3-mom)"]
SLIM_COLORS     = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]


# ── Bootstrap core ────────────────────────────────────────────────────────────

def bootstrap_delta_auc(labels, baseline_scores, ablated_scores, n_bootstrap=1000, seed=42):
    """
    Paired bootstrap: resample jet indices with replacement, compute
    ΔAUC = AUC(baseline) - AUC(ablated) on each resample.

    Returns
    -------
    delta_mean : float
    ci_lo, ci_hi : float  (2.5th and 97.5th percentile)
    p_value : float  (one-sided: fraction of resamples where ΔAUC ≤ 0)
    """
    n = len(labels)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        auc_base = roc_auc_score(labels[idx], baseline_scores[idx])
        auc_abl  = roc_auc_score(labels[idx], ablated_scores[idx])
        deltas[b] = auc_base - auc_abl

    return (
        float(np.mean(deltas)),
        float(np.percentile(deltas,  2.5)),
        float(np.percentile(deltas, 97.5)),
        float(np.mean(deltas <= 0)),   # p-value: fraction where ΔAUC ≤ 0
    )


def hanley_mcneil_se(auc, n_pos, n_neg):
    """
    Hanley & McNeil (1982) closed-form SE for AUC.
    n_pos / n_neg: number of positive / negative examples.
    """
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2)
           + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)
    return float(np.sqrt(max(var, 0)))


# ── LGATr analysis ────────────────────────────────────────────────────────────

def analyse_lgatr(scores_path, output_dir, n_bootstrap, seed):
    d = np.load(scores_path)
    labels   = d["labels"].astype(np.float32)
    baseline = d["baseline"].astype(np.float32)
    n_jets   = len(labels)

    n_pos = int(labels.sum())
    n_neg = n_jets - n_pos
    base_auc = float(roc_auc_score(labels, baseline))

    print(f"\n{'='*65}")
    print(f"L-GATr  grade ablation significance  ({n_jets} jets, AUC={base_auc:.5f})")
    print(f"{'='*65}")
    print(f"{'Ablation':<22}  {'AUC_abl':>8}  {'ΔAUC':>7}  "
          f"{'CI_lo':>7}  {'CI_hi':>7}  {'p-val':>7}")
    print("-" * 65)

    zero_means, zero_lo, zero_hi = [], [], []
    keep_means, keep_lo, keep_hi = [], [], []

    for g in range(5):
        # Zero-grade ablation
        key_z = f"zero_g{g}"
        if key_z in d:
            abl_s = d[key_z].astype(np.float32)
            auc_a = float(roc_auc_score(labels, abl_s))
            dm, lo, hi, pv = bootstrap_delta_auc(labels, baseline, abl_s, n_bootstrap, seed + g)
            zero_means.append(dm); zero_lo.append(lo); zero_hi.append(hi)
            print(f"  zero {GRADE_NAMES[g]:<16}  {auc_a:.5f}  {dm:>+7.4f}  "
                  f"{lo:>7.4f}  {hi:>7.4f}  {pv:>7.4f}")
        else:
            zero_means.append(np.nan); zero_lo.append(np.nan); zero_hi.append(np.nan)
            print(f"  zero {GRADE_NAMES[g]:<16}  (no scores saved)")

        # Keep-only ablation
        key_k = f"keep_g{g}"
        if key_k in d:
            abl_s = d[key_k].astype(np.float32)
            auc_a = float(roc_auc_score(labels, abl_s))
            dm, lo, hi, pv = bootstrap_delta_auc(labels, baseline, abl_s, n_bootstrap, seed + 10 + g)
            keep_means.append(dm); keep_lo.append(lo); keep_hi.append(hi)
            print(f"  keep {GRADE_NAMES[g]:<16}  {auc_a:.5f}  {dm:>+7.4f}  "
                  f"{lo:>7.4f}  {hi:>7.4f}  {pv:>7.4f}")
        else:
            keep_means.append(np.nan); keep_lo.append(np.nan); keep_hi.append(np.nan)
            print(f"  keep {GRADE_NAMES[g]:<16}  (no scores saved)")

    # Load existing summary AUC values for bar chart
    summary_path = Path(scores_path).parent / "grade_decomp_full.npz"
    if summary_path.exists():
        ds = np.load(summary_path)
        grade_aucs     = ds["grade_aucs"]
        keep_only_aucs = ds["keep_only_aucs"]
        _plot_lgatr_with_ci(base_auc, grade_aucs, keep_only_aucs,
                            np.array(zero_means), np.array(zero_lo), np.array(zero_hi),
                            np.array(keep_means), np.array(keep_lo), np.array(keep_hi),
                            n_jets, n_bootstrap,
                            output_dir / "grade_decomp_full_ci.png")

    # Save bootstrap results
    np.savez(
        output_dir / "grade_decomp_bootstrap_ci.npz",
        zero_delta_mean=np.array(zero_means), zero_delta_lo=np.array(zero_lo),
        zero_delta_hi=np.array(zero_hi),
        keep_delta_mean=np.array(keep_means), keep_delta_lo=np.array(keep_lo),
        keep_delta_hi=np.array(keep_hi),
        n_bootstrap=np.int32(n_bootstrap),
    )
    print(f"Saved CI arrays → {output_dir / 'grade_decomp_bootstrap_ci.npz'}")


def _plot_lgatr_with_ci(baseline_auc, grade_aucs, keep_only_aucs,
                        zero_dm, zero_lo, zero_hi,
                        keep_dm, keep_lo, keep_hi,
                        n_jets, n_bootstrap, save_path):
    x = np.arange(5)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: zero-grade ablation
    ax = axes[0]
    bars = ax.bar(x, grade_aucs, width=0.6, color=GRADE_COLORS, edgecolor="k", alpha=0.85)
    ax.axhline(baseline_auc, color="black", linestyle="--", linewidth=1.5,
               label=f"Baseline {baseline_auc:.4f}")
    # Error bars from bootstrap CI (on ΔAUC → AUC = baseline - delta)
    valid = ~np.isnan(zero_dm)
    if valid.any():
        auc_from_boot = baseline_auc - zero_dm
        err_lo = auc_from_boot - (baseline_auc - zero_hi)  # CI lower bound in AUC space
        err_hi = (baseline_auc - zero_lo) - auc_from_boot  # CI upper bound in AUC space
        ax.errorbar(x[valid], auc_from_boot[valid],
                    yerr=[err_lo[valid], err_hi[valid]],
                    fmt="none", color="black", capsize=5, linewidth=1.5,
                    label=f"95% CI (B={n_bootstrap})")
    for bar, val in zip(bars, grade_aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    ymin = max(0.4, float(grade_aucs.min()) - 0.02)
    ax.set_ylim(ymin, baseline_auc + 0.01)
    ax.set_xticks(x)
    ax.set_xticklabels(GRADE_NAMES, rotation=20, ha="right")
    ax.set_ylabel("AUC", fontsize=12)
    ax.set_title("Zero-grade ablation (all layers)\nHow necessary is each grade?", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Right: keep-only ablation
    ax = axes[1]
    bars = ax.bar(x, keep_only_aucs, width=0.6, color=GRADE_COLORS, edgecolor="k", alpha=0.85)
    ax.axhline(baseline_auc, color="black", linestyle="--", linewidth=1.5,
               label=f"Baseline {baseline_auc:.4f}")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.0, label="Random (0.5)")
    valid = ~np.isnan(keep_dm)
    if valid.any():
        auc_from_boot = baseline_auc - keep_dm
        err_lo = auc_from_boot - (baseline_auc - keep_hi)
        err_hi = (baseline_auc - keep_lo) - auc_from_boot
        ax.errorbar(x[valid], auc_from_boot[valid],
                    yerr=[err_lo[valid], err_hi[valid]],
                    fmt="none", color="black", capsize=5, linewidth=1.5,
                    label=f"95% CI (B={n_bootstrap})")
    for bar, val in zip(bars, keep_only_aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0.45, baseline_auc + 0.01)
    ax.set_xticks(x)
    ax.set_xticklabels(GRADE_NAMES, rotation=20, ha="right")
    ax.set_ylabel("AUC", fontsize=12)
    ax.set_title("Keep-only (all other grades zeroed)\nHow sufficient is each grade alone?",
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"L-GATr Grade Decomposition  —  TopTagging  ({n_jets} jets)\n"
        f"Baseline AUC = {baseline_auc:.4f}  |  error bars = 95% bootstrap CI (B={n_bootstrap})",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")


# ── LGATr-slim analysis ───────────────────────────────────────────────────────

def analyse_slim(scores_path, output_dir, n_bootstrap, seed):
    d = np.load(scores_path)
    labels   = d["labels"].astype(np.float32)
    baseline = d["baseline"].astype(np.float32)
    n_jets   = len(labels)

    base_auc = float(roc_auc_score(labels, baseline))

    print(f"\n{'='*65}")
    print(f"L-GATr-slim  grade ablation significance  ({n_jets} jets, AUC={base_auc:.5f})")
    print(f"{'='*65}")
    print(f"{'Ablation':<22}  {'AUC_abl':>8}  {'ΔAUC':>7}  "
          f"{'CI_lo':>7}  {'CI_hi':>7}  {'p-val':>7}")
    print("-" * 65)

    abl_means, abl_lo_arr, abl_hi_arr = [], [], []

    for i, abl_name in enumerate(SLIM_ABL_NAMES):
        if abl_name not in d:
            abl_means.append(np.nan); abl_lo_arr.append(np.nan); abl_hi_arr.append(np.nan)
            print(f"  {abl_name:<20}  (no scores saved)")
            continue
        abl_s = d[abl_name].astype(np.float32)
        auc_a = float(roc_auc_score(labels, abl_s))
        dm, lo, hi, pv = bootstrap_delta_auc(labels, baseline, abl_s, n_bootstrap, seed + 20 + i)
        abl_means.append(dm); abl_lo_arr.append(lo); abl_hi_arr.append(hi)
        print(f"  {abl_name:<20}  {auc_a:.5f}  {dm:>+7.4f}  "
              f"{lo:>7.4f}  {hi:>7.4f}  {pv:>7.4f}")

    # Load existing AUC values for bar chart
    summary_path = Path(scores_path).parent / "grade_decomp_slim.npz"
    if summary_path.exists():
        ds = np.load(summary_path)
        abl_aucs = ds["abl_aucs"]
        _plot_slim_with_ci(base_auc, abl_aucs,
                           np.array(abl_means), np.array(abl_lo_arr), np.array(abl_hi_arr),
                           n_jets, n_bootstrap,
                           output_dir / "grade_decomp_slim_full_ci.png")

    np.savez(
        output_dir / "grade_decomp_slim_bootstrap_ci.npz",
        abl_delta_mean=np.array(abl_means),
        abl_delta_lo=np.array(abl_lo_arr),
        abl_delta_hi=np.array(abl_hi_arr),
        n_bootstrap=np.int32(n_bootstrap),
    )
    print(f"Saved CI arrays → {output_dir / 'grade_decomp_slim_bootstrap_ci.npz'}")


def _plot_slim_with_ci(baseline_auc, abl_aucs, abl_dm, abl_lo, abl_hi,
                       n_jets, n_bootstrap, save_path):
    x = np.arange(len(SLIM_ABL_NAMES))
    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(x, abl_aucs, width=0.6, color=SLIM_COLORS, edgecolor="k", alpha=0.85)
    ax.axhline(baseline_auc, color="black", linestyle="--", linewidth=1.5,
               label=f"Baseline {baseline_auc:.4f}")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.0, label="Random (0.5)")

    valid = ~np.isnan(abl_dm)
    if valid.any():
        auc_from_boot = baseline_auc - abl_dm
        err_lo = auc_from_boot - (baseline_auc - abl_hi)
        err_hi = (baseline_auc - abl_lo) - auc_from_boot
        ax.errorbar(x[valid], auc_from_boot[valid],
                    yerr=[err_lo[valid], err_hi[valid]],
                    fmt="none", color="black", capsize=6, linewidth=1.5,
                    label=f"95% CI (B={n_bootstrap})")

    for bar, val in zip(bars, abl_aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylim(0.45, baseline_auc + 0.01)
    ax.set_xticks(x)
    ax.set_xticklabels(SLIM_ABL_LABELS, fontsize=9)
    ax.set_ylabel("AUC", fontsize=12)
    ax.set_title(
        f"L-GATr-slim Channel Ablation  —  TopTagging  ({n_jets} jets)\n"
        f"Baseline AUC = {baseline_auc:.4f}  |  error bars = 95% bootstrap CI (B={n_bootstrap})",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")


# ── LGATr workshop (3-intervention) analysis ─────────────────────────────────

WORKSHOP_KEYS   = ["scalar_like", "vector_like", "bivector"]
WORKSHOP_LABELS = ["No scalar/pseudoscalar\n(G0+G4)", "No vector/trivector\n(G1+G3)", "No bivector\n(G2)"]
WORKSHOP_COLORS = ["#4C72B0", "#DD8452", "#55A868"]


def analyse_lgatr_workshop(scores_path, output_dir, n_bootstrap, seed, label="L-GATr"):
    d = np.load(scores_path)
    labels   = d["labels"].astype(np.float32)
    baseline = d["baseline"].astype(np.float32)
    n_jets   = len(labels)

    n_pos = int(labels.sum())
    n_neg = n_jets - n_pos
    base_auc = float(roc_auc_score(labels, baseline))

    print(f"\n{'='*65}")
    print(f"{label}  workshop-grade bootstrap  ({n_jets} jets, AUC={base_auc:.5f})")
    print(f"{'='*65}")
    print(f"{'Ablation':<28}  {'AUC_abl':>8}  {'ΔAUC':>7}  {'CI_lo':>7}  {'CI_hi':>7}  {'p-val':>7}")
    print("-" * 65)

    zero_means, zero_lo, zero_hi = [], [], []
    keep_means, keep_lo, keep_hi = [], [], []

    for i, key in enumerate(WORKSHOP_KEYS):
        for kind, dm_list, lo_list, hi_list in [
            ("zero", zero_means, zero_lo, zero_hi),
            ("keep", keep_means, keep_lo, keep_hi),
        ]:
            full_key = f"{kind}_{key}"
            if full_key not in d:
                dm_list.append(np.nan); lo_list.append(np.nan); hi_list.append(np.nan)
                print(f"  {kind} {key:<22}  (not saved)")
                continue
            abl_s = d[full_key].astype(np.float32)
            auc_a = float(roc_auc_score(labels, abl_s))
            dm, lo, hi, pv = bootstrap_delta_auc(labels, baseline, abl_s, n_bootstrap, seed + i * 2 + (0 if kind == "zero" else 1))
            dm_list.append(dm); lo_list.append(lo); hi_list.append(hi)
            print(f"  {kind} {key:<22}  {auc_a:.5f}  {dm:>+7.4f}  {lo:>7.4f}  {hi:>7.4f}  {pv:>7.4f}")

    summary_path = Path(scores_path).parent / "grade_decomp_full.npz"
    if summary_path.exists():
        ds = np.load(summary_path)
        grade_aucs     = ds.get("grade_aucs",     None)
        keep_only_aucs = ds.get("keep_only_aucs", None)
        if grade_aucs is not None and keep_only_aucs is not None:
            _plot_workshop_with_ci(
                base_auc, grade_aucs, keep_only_aucs,
                np.array(zero_means), np.array(zero_lo), np.array(zero_hi),
                np.array(keep_means), np.array(keep_lo), np.array(keep_hi),
                n_jets, n_bootstrap, label,
                output_dir / "grade_decomp_full_ci.png",
            )

    np.savez(
        output_dir / "grade_decomp_workshop_bootstrap_ci.npz",
        zero_delta_mean=np.array(zero_means), zero_delta_lo=np.array(zero_lo),
        zero_delta_hi=np.array(zero_hi),
        keep_delta_mean=np.array(keep_means), keep_delta_lo=np.array(keep_lo),
        keep_delta_hi=np.array(keep_hi),
        keys=np.array(WORKSHOP_KEYS),
        n_bootstrap=np.int32(n_bootstrap),
    )
    print(f"Saved CI arrays -> {output_dir / 'grade_decomp_workshop_bootstrap_ci.npz'}")


def _plot_workshop_with_ci(baseline_auc, grade_aucs, keep_only_aucs,
                           zero_dm, zero_lo, zero_hi,
                           keep_dm, keep_lo, keep_hi,
                           n_jets, n_bootstrap, label, save_path):
    n = len(WORKSHOP_KEYS)
    x = np.arange(n)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, aucs, dm, lo, hi, title_suffix in [
        (axes[0], grade_aucs,     zero_dm, zero_lo, zero_hi, "Zero-grade (how necessary?)"),
        (axes[1], keep_only_aucs, keep_dm, keep_lo, keep_hi, "Keep-only (how sufficient?)"),
    ]:
        bars = ax.bar(x, aucs[:n], width=0.6, color=WORKSHOP_COLORS, edgecolor="k", alpha=0.85)
        ax.axhline(baseline_auc, color="black", linestyle="--", linewidth=1.5,
                   label=f"Baseline {baseline_auc:.4f}")
        valid = ~np.isnan(dm)
        if valid.any():
            auc_boot = baseline_auc - dm
            ax.errorbar(x[valid], auc_boot[valid],
                        yerr=[auc_boot[valid] - (baseline_auc - hi[valid]),
                              (baseline_auc - lo[valid]) - auc_boot[valid]],
                        fmt="none", color="black", capsize=5, linewidth=1.5,
                        label=f"95% CI (B={n_bootstrap})")
        for bar, val in zip(bars, aucs[:n]):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.001,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9)
        ymin = max(0.4, float(aucs[:n].min()) - 0.02)
        ax.set_ylim(ymin, baseline_auc + 0.01)
        ax.set_xticks(x)
        ax.set_xticklabels(WORKSHOP_LABELS, fontsize=9)
        ax.set_ylabel("AUC", fontsize=12)
        ax.set_title(title_suffix, fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"{label}  Grade Decomposition  —  TopTagging  ({n_jets} jets)\n"
        f"Baseline AUC = {baseline_auc:.4f}  |  error bars = 95% bootstrap CI (B={n_bootstrap})",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {save_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores",   required=True,
                    help="Path to grade_decomp_scores.npz or grade_decomp_slim_scores.npz")
    ap.add_argument("--mode",     required=True,
                    choices=["all_grades", "workshop", "slim"],
                    help="all_grades: subgroup_false (G0-G4); workshop: baseline/bivector_false (G0+G4,G1+G3,G2); slim: LGATr-slim")
    ap.add_argument("--output",   required=True, help="Output directory")
    ap.add_argument("--label",    default=None,  help="Model label for plot titles")
    ap.add_argument("--n-bootstrap",  type=int, default=1000,
                    help="Bootstrap resamples")
    ap.add_argument("--seed",         type=int, default=42)
    args = ap.parse_args()

    p = Path(args.scores)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if not p.exists():
        print(f"ERROR: {p} not found.")
        return

    label = args.label or args.mode

    if args.mode == "all_grades":
        analyse_lgatr(p, out, args.n_bootstrap, args.seed)
    elif args.mode == "workshop":
        analyse_lgatr_workshop(p, out, args.n_bootstrap, args.seed, label=label)
    elif args.mode == "slim":
        analyse_slim(p, out, args.n_bootstrap, args.seed)


if __name__ == "__main__":
    main()
