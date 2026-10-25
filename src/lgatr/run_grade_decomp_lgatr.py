#!/usr/bin/env python3
"""Grade ablations on L-GATr: zero or keep one grade group of the hidden h_mv and measure the test AUC.

h_mv components by grade: 0 (index 0), 1 (1-4), 2 (5-10), 3 (11-14), 4 (15). With
--interventions workshop (the default connected-subgroup model) the groups are G0+G4, G1+G3 and
G2; with all_grades the five grades are separate (the model trained with
use_fully_connected_subgroup=False). Ablations act on the hidden outputs of linear_in and every
block, not on the input embedding.

  --mode full    zero one group at all layers, and keep only one group at all layers
  --mode layer   zero one group at a single layer, for every layer

Outputs: grade_decomp_full.npz, grade_decomp_layer.npz, with --save-scores the per-jet logits
grade_decomp_scores.npz used by src/uncertainty/compute_combined_bootstrap_ci.py.

    python src/lgatr/run_grade_decomp_lgatr.py --checkpoint $RUNS/topt_lgatr/seed1001/models/model_run0.pt \\
        --config $RUNS/topt_lgatr/seed1001/config.yaml --data-path $DATA --n-jets 10000 --batch-size 64 \\
        --mode full --interventions workshop --include-clean-scalar-side-study \\
        --include-clean-keep-only-side-study --save-scores --output-dir results/ablations/lgatr/seed1001
    python src/lgatr/run_grade_decomp_lgatr.py ... --mode layer --interventions workshop \\
        --output-dir results/ablations/lgatr_layer/seed1001
"""

import argparse
import sys
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:  # optional; only needed for heatmaps
    sns = None
from pathlib import Path
from omegaconf import OmegaConf
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")

from experiments.tagging.embedding import embed_tagging_data

# ── Grade structure of Cl(1,3) 16-component multivector ──────────────────────
# These are slices into the LAST dimension of h_mv (the GA basis dimension).
GRADE_SLICES = [
    slice(0, 1),    # grade 0: scalar        (1 component)
    slice(1, 5),    # grade 1: vectors       (4 components)
    slice(5, 11),   # grade 2: bivectors     (6 components)
    slice(11, 15),  # grade 3: trivectors    (4 components)
    slice(15, 16),  # grade 4: pseudoscalar  (1 component)
]
GRADE_NAMES = [
    "G0 scalar",
    "G1 vector",
    "G2 bivector",
    "G3 trivector",
    "G4 pseudoscalar",
]
GRADE_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

GRADE_INTERVENTIONS = [
    {
        "key": f"g{idx}",
        "name": name,
        "plot_name": name,
        "slices": [GRADE_SLICES[idx]],
        "color": GRADE_COLORS[idx],
        "grade_mask": [1 if j == idx else 0 for j in range(5)],
    }
    for idx, name in enumerate(GRADE_NAMES)
]

# Grade groups under the connected subgroup:
#   no-scalar:  zero stored G0 + G4
#   no-vector:  zero stored G1 + G3
#   no-bivector: zero stored G2
WORKSHOP_INTERVENTIONS = [
    {
        "key": "scalar_like",
        "name": "G0+G4 scalar/pseudoscalar",
        "plot_name": "No scalar/pseudoscalar",
        "slices": [GRADE_SLICES[0], GRADE_SLICES[4]],
        "color": "#4C72B0",
        "grade_mask": [1, 0, 0, 0, 1],
    },
    {
        "key": "vector_like",
        "name": "G1+G3 vector/trivector",
        "plot_name": "No vector/trivector",
        "slices": [GRADE_SLICES[1], GRADE_SLICES[3]],
        "color": "#DD8452",
        "grade_mask": [0, 1, 0, 1, 0],
    },
    {
        "key": "bivector",
        "name": "G2 bivector",
        "plot_name": "No bivector",
        "slices": [GRADE_SLICES[2]],
        "color": "#55A868",
        "grade_mask": [0, 0, 1, 0, 0],
    },
]

# Scalar-off diagnostic:
# zero stored G0+G4 *and* the separate hidden scalar branch h_s.
CLEAN_SCALAR_SIDE_STUDY = {
    "key": "scalar_total",
    "name": "G0+G4 plus scalar branch",
    "plot_name": "No scalar total",
    "slices": [GRADE_SLICES[0], GRADE_SLICES[4]],
    "color": "#4C72B0",
    "grade_mask": [1, 0, 0, 0, 1],
    "zero_hidden_scalars": True,
}

CLEAN_KEEP_ONLY_SIDE_STUDIES = [
    {
        "key": "vector_total",
        "name": "G1+G3 only, no scalar branch",
        "plot_name": "Only vector/trivector",
        "slices": [GRADE_SLICES[1], GRADE_SLICES[3]],
        "color": "#DD8452",
        "grade_mask": [0, 1, 0, 1, 0],
        "zero_hidden_scalars": True,
    },
    {
        "key": "bivector_total",
        "name": "G2 only, no scalar branch",
        "plot_name": "Only bivector",
        "slices": [GRADE_SLICES[2]],
        "color": "#55A868",
        "grade_mask": [0, 0, 1, 0, 0],
        "zero_hidden_scalars": True,
    },
]

NUM_LAYERS = 13   # linear_in (L0) + 12 blocks (L1..L12)
EPS = 1e-5


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(checkpoint_path, config_path, device):
    from hydra.utils import instantiate
    cfg = OmegaConf.load(config_path)
    if "compile" in cfg.model.net:
        del cfg.model.net["compile"]
    if device.type == "cpu":
        cfg.model.attention_backend = "math"
    if hasattr(cfg, "ga_settings"):
        from lgatr.primitives.config import gatr_config
        gatr_config.use_fully_connected_subgroup = bool(cfg.ga_settings.use_fully_connected_subgroup)
        gatr_config.use_bivector = bool(cfg.ga_settings.use_bivector)
        gatr_config.use_geometric_product = bool(cfg.ga_settings.use_geometric_product)
        print(
            f"Applied ga_settings: use_fully_connected_subgroup={gatr_config.use_fully_connected_subgroup}, "
            f"use_bivector={gatr_config.use_bivector}, "
            f"use_geometric_product={gatr_config.use_geometric_product}"
        )
    model = instantiate(cfg.model)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["model"]
    if any(k.startswith("net.module.") for k in state):
        state = {k.replace("net.module.", "net.", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"Loaded model from {checkpoint_path}")
    return model, cfg


def build_toptagging_loader(data_path, n_jets, batch_size, momentum_float64):
    """Build a DataLoader from only the leading n_jets test events."""
    momentum_dtype = torch.float64 if momentum_float64 else torch.float32
    data = np.load(data_path, mmap_mode="r")
    kinematics = data["kinematics_test"][:n_jets]
    labels = data["labels_test"][:n_jets]

    data_list = []
    for i in range(kinematics.shape[0]):
        fourmomenta = torch.tensor(kinematics[i], dtype=momentum_dtype)
        mask = (fourmomenta.abs() > EPS).all(dim=-1)
        fourmomenta = fourmomenta[mask]
        scalars = torch.zeros(fourmomenta.shape[0], 0, dtype=torch.float32)
        label = torch.tensor(labels[i], dtype=torch.bool)
        data_list.append(Data(x=fourmomenta, scalars=scalars, label=label))

    return DataLoader(data_list, batch_size=batch_size, shuffle=False)


# ── Hook factory ─────────────────────────────────────────────────────────────

def make_zero_intervention_hook(intervention):
    """
    Returns a forward hook that zeros one intervention group in h_mv.

    Hooks fire on modules whose output is (h_mv, h_s):
      - model.net.linear_in
      - model.net.blocks[i]   (GATrBlock)

    h_mv shape inside the hook: (1, N_total_tokens, hidden_mv_channels=16, 16)
    The last dimension is the GA basis; we zero the target grade slice(s).
    """
    slices = intervention["slices"]
    def hook(module, inp, out):
        h_mv, h_s = out
        h_mv = h_mv.clone()
        for s in slices:
            h_mv[..., s] = 0.0
        if intervention.get("zero_hidden_scalars", False) and h_s is not None:
            h_s = torch.zeros_like(h_s)
        return (h_mv, h_s)
    return hook


def make_keep_only_intervention_hook(intervention):
    """
    Returns a forward hook that zeros ALL grades EXCEPT one intervention group.
    This is the complement of make_zero_intervention_hook.
    """
    keep_ids = {idx for idx, active in enumerate(intervention["grade_mask"]) if active}
    other_slices = [GRADE_SLICES[g] for g in range(5) if g not in keep_ids]
    def hook(module, inp, out):
        h_mv, h_s = out
        h_mv = h_mv.clone()
        for s in other_slices:
            h_mv[..., s] = 0.0
        if intervention.get("zero_hidden_scalars", False) and h_s is not None:
            h_s = torch.zeros_like(h_s)
        return (h_mv, h_s)
    return hook


def register_full_network_hooks(model, intervention):
    """Register intervention-zeroing hooks on ALL 13 layers. Returns handles."""
    hook = make_zero_intervention_hook(intervention)
    handles = []
    handles.append(model.net.linear_in.register_forward_hook(hook))
    for blk in model.net.blocks:
        handles.append(blk.register_forward_hook(hook))
    return handles


def register_keep_only_hooks(model, intervention):
    """Register keep-only hooks on ALL 13 layers. Returns list of handles."""
    hook = make_keep_only_intervention_hook(intervention)
    handles = []
    handles.append(model.net.linear_in.register_forward_hook(hook))
    for blk in model.net.blocks:
        handles.append(blk.register_forward_hook(hook))
    return handles


def register_single_layer_hook(model, intervention, layer_idx):
    """
    Register intervention-zeroing hook on ONE layer only.
    layer_idx 0 → linear_in; layer_idx 1..12 → blocks[layer_idx-1].
    Returns list with a single handle.
    """
    hook = make_zero_intervention_hook(intervention)
    if layer_idx == 0:
        return [model.net.linear_in.register_forward_hook(hook)]
    else:
        return [model.net.blocks[layer_idx - 1].register_forward_hook(hook)]


# ── Inference helper ──────────────────────────────────────────────────────────

def run_inference(model, loader, cfg, n_jets, device, handles=()):
    """
    Run model.forward over up to n_jets jets (in dataset order).
    hooks_handles: list of already-registered hook handles (will NOT be removed here).

    Returns (labels: np.array, scores: np.array) — both shape (n_jets,).
    Scores are the raw logit (higher = more top-like).
    """
    all_logits, all_labels = [], []
    n_done = 0

    with torch.no_grad():
        for batch in loader:
            if n_done >= n_jets:
                break

            batch_dev = batch.to(device)
            scalars = (
                batch_dev.scalars
                if (hasattr(batch_dev, "scalars") and batch_dev.scalars is not None)
                else torch.zeros(batch_dev.x.shape[0], 0, device=device)
            )
            embedding = embed_tagging_data(batch_dev.x, scalars, batch_dev.ptr, cfg.data)

            result = model(embedding)
            logits = result[0]   # LGATrWrapper returns (logits, tracker, frames)

            B    = logits.shape[0]
            take = min(B, n_jets - n_done)
            all_logits.extend(logits[:take, 0].cpu().tolist())
            all_labels.extend(batch.label[:take].cpu().tolist())
            n_done += take

    return np.array(all_labels, dtype=np.float32), np.array(all_logits, dtype=np.float32)


# ── Plotting ──────────────────────────────────────────────────────────────────

def _names(interventions):
    return [item["plot_name"] for item in interventions]


def _colors(interventions):
    return [item["color"] for item in interventions]


def plot_full_network(baseline_auc, grade_aucs, keep_only_aucs, interventions, n_jets, save_path):
    """
    Two-panel plot:
      Left:  AUC when each grade is zeroed (ablation) vs baseline — shows necessity
      Right: AUC when ONLY that grade is kept — shows sufficiency; grade-1 directly
             comparable to LGATr-slim (pure vector architecture)
    """
    names = _names(interventions)
    colors = _colors(interventions)
    x = np.arange(len(interventions))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ── Left: ablation (zero one grade at all layers) ────────────────────────
    ax = axes[0]
    bars = ax.bar(x, grade_aucs, width=0.6, color=colors, edgecolor="k", alpha=0.85)
    ax.axhline(baseline_auc, color="black", linestyle="--", linewidth=1.5,
               label=f"Baseline {baseline_auc:.4f}")
    for bar, val in zip(bars, grade_aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0003,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    ymin = max(0.4, grade_aucs.min() - 0.02)
    ax.set_ylim(ymin, baseline_auc + 0.008)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("AUC", fontsize=12)
    ax.set_title("Zero intervention (all layers)\nHow necessary is each component group?", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # ── Right: keep-only (zero all other grades at all layers) ───────────────
    ax = axes[1]
    bars = ax.bar(x, keep_only_aucs, width=0.6, color=colors, edgecolor="k", alpha=0.85)
    ax.axhline(baseline_auc, color="black", linestyle="--", linewidth=1.5,
               label=f"Baseline {baseline_auc:.4f}")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.0, label="Random (0.5)")
    for bar, val in zip(bars, keep_only_aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0.45, baseline_auc + 0.008)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("AUC", fontsize=12)
    ax.set_title("Keep-only (all other grades zeroed)\nHow sufficient is each grade alone?",
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"L-GATr Connected-Subgroup Ablations  —  TopTagging  ({n_jets} jets)\n"
        f"Baseline AUC = {baseline_auc:.4f}",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved full-network grade plot → {save_path}")


def plot_layer_resolved(baseline_auc, layer_aucs, interventions, n_jets, save_path):
    """Heatmap of ΔAUC per (intervention × layer) for layer-resolved ablation."""
    delta = baseline_auc - layer_aucs   # shape (5, 13)

    layer_labels = ["L0"] + [f"L{i}" for i in range(1, NUM_LAYERS)]

    fig, ax = plt.subplots(figsize=(14, 5))
    if sns is not None:
        sns.heatmap(
            delta,
            ax=ax,
            cmap="Reds",
            vmin=0,
            vmax=max(delta.max(), 1e-4),
            annot=True,
            fmt=".4f",
            annot_kws={"size": 7},
            xticklabels=layer_labels,
            yticklabels=_names(interventions),
            linewidths=0.3,
        )
    else:
        im = ax.imshow(delta, aspect="auto", cmap="Reds", vmin=0, vmax=max(delta.max(), 1e-4))
        ax.set_xticks(np.arange(len(layer_labels)))
        ax.set_xticklabels(layer_labels)
        ax.set_yticks(np.arange(len(interventions)))
        ax.set_yticklabels(_names(interventions))
        for row in range(delta.shape[0]):
            for col in range(delta.shape[1]):
                ax.text(col, row, f"{delta[row, col]:.4f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("Layer (ablation applied at this layer's output)", fontsize=11)
    ax.set_ylabel("Grade zeroed", fontsize=11)
    ax.set_title(
        f"L-GATr Layer-Resolved Connected-Subgroup Ablation (TopTagging, {n_jets} jets)\n"
        f"ΔAUC = baseline ({baseline_auc:.4f}) − ablated  |  brighter = more important",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved layer-resolved grade heatmap → {save_path}")


def write_summary(
    baseline_auc,
    grade_aucs,
    keep_only_aucs,
    layer_aucs,
    interventions,
    n_jets,
    save_path,
    extra_zero_only_results=None,
    extra_keep_only_results=None,
):
    lines = []
    lines.append(f"Connected-Subgroup Ablation Summary  —  L-GATr (TopTagging, {n_jets} jets)")
    lines.append("=" * 75)
    lines.append(f"\nBaseline AUC (no ablation): {baseline_auc:.5f}\n")

    if grade_aucs is not None:
        lines.append("Full-network ablation (component group zeroed at ALL layers)")
        lines.append("-" * 60)
        lines.append(f"{'Intervention':<30}  {'AUC ablated':>12}  {'ΔAUC':>8}")
        for row, item in enumerate(interventions):
            lines.append(
                f"  {item['name']:<28}  {grade_aucs[row]:>12.5f}  {baseline_auc - grade_aucs[row]:>+8.5f}"
            )

    if keep_only_aucs is not None:
        lines.append("\nKeep-only ablation (all OTHER component groups zeroed at ALL layers)")
        lines.append("-" * 60)
        lines.append(f"{'Intervention':<30}  {'AUC keep-only':>13}  {'vs baseline':>12}")
        for row, item in enumerate(interventions):
            lines.append(
                f"  {item['name']:<28}  {keep_only_aucs[row]:>13.5f}  {baseline_auc - keep_only_aucs[row]:>+12.5f}"
            )
        lines.append(
            "\n  Note: default workshop interventions follow the connected-subgroup setup: "
            "G0+G4, G1+G3, and G2."
        )

    if layer_aucs is not None:
        lines.append("\nLayer-resolved ablation (component group zeroed at ONE layer output)")
        lines.append("-" * 60)
        lines.append("ΔAUC values  (rows=interventions, cols=layers L0..L12)")
        header = f"{'Intervention':<30}" + "".join(f" {f'L{i}':>7}" for i in range(NUM_LAYERS))
        lines.append(header)
        for row_idx, item in enumerate(interventions):
            row = f"  {item['name']:<28}"
            for l in range(NUM_LAYERS):
                row += f" {baseline_auc - layer_aucs[row_idx, l]:>+7.4f}"
            lines.append(row)

    if extra_zero_only_results:
        lines.append("\nAppendix scalar side-study (zero-only)")
        lines.append("-" * 60)
        lines.append(f"{'Intervention':<30}  {'AUC ablated':>12}  {'ΔAUC':>8}")
        for item, auc_val in extra_zero_only_results:
            lines.append(
                f"  {item['name']:<28}  {auc_val:>12.5f}  {baseline_auc - auc_val:>+8.5f}"
            )
        lines.append(
            "\n  This post-hoc diagnostic zeros both stored G0+G4 multivector components "
            "and the separate hidden scalar branch h_s."
        )

    if extra_keep_only_results:
        lines.append("\nAppendix strict keep-only side-study")
        lines.append("-" * 60)
        lines.append(f"{'Intervention':<30}  {'AUC keep-only':>13}  {'vs baseline':>12}")
        for item, auc_val in extra_keep_only_results:
            lines.append(
                f"  {item['name']:<28}  {auc_val:>13.5f}  {baseline_auc - auc_val:>+12.5f}"
            )
        lines.append(
            "\n  These post-hoc diagnostics keep only the selected multivector sector and "
            "also zero the separate hidden scalar branch h_s."
        )

    text = "\n".join(lines)
    save_path.write_text(text)
    print(f"\nSaved summary → {save_path}")
    print(text)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Grade decomposition ablation for L-GATr on TopTagging"
    )
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config",     required=True)
    ap.add_argument("--data-path",  default="data/toptagging_full.npz")
    ap.add_argument("--output-dir", default="results/grade_decomp")
    ap.add_argument(
        "--mode", choices=["full", "layer", "all"], default="all",
        help=(
            "full: baseline + full-network ablation for selected interventions; "
            "layer: layer-resolved ablation for selected interventions; "
            "all: both"
        ),
    )
    ap.add_argument(
        "--interventions", choices=["workshop", "all_grades"], default="workshop",
        help=(
            "workshop: connected-subgroup groups G0+G4, G1+G3, G2; "
            "all_grades: five separate grades, for the model trained with use_fully_connected_subgroup=False"
        ),
    )
    ap.add_argument(
        "--save-scores",
        action="store_true",
        help="Save per-jet baseline and intervention logits for paired bootstrap.",
    )
    ap.add_argument(
        "--include-clean-scalar-side-study",
        action="store_true",
        help=(
            "Also run the scalar-off diagnostic: zero stored G0+G4 "
            "and the separate hidden scalar branch h_s at every layer."
        ),
    )
    ap.add_argument(
        "--include-clean-keep-only-side-study",
        action="store_true",
        help=(
            "Also run strict appendix keep-only diagnostics where the selected multivector "
            "sector is kept and the separate hidden scalar branch h_s is zeroed."
        ),
    )
    ap.add_argument("--n-jets",     type=int, default=10000,
                    help="Jets to evaluate per run (10000 matches other experiments)")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Mode: {args.mode}  |  n_jets={args.n_jets}  |  batch_size={args.batch_size}")
    interventions = WORKSHOP_INTERVENTIONS if args.interventions == "workshop" else GRADE_INTERVENTIONS
    print("Interventions:", ", ".join(item["name"] for item in interventions))
    if args.include_clean_scalar_side_study:
        print("Extra zero-only side-study:", CLEAN_SCALAR_SIDE_STUDY["name"])
    if args.include_clean_keep_only_side_study:
        print(
            "Extra keep-only side-study:",
            ", ".join(item["name"] for item in CLEAN_KEEP_ONLY_SIDE_STUDIES),
        )

    model, cfg = load_model(args.checkpoint, args.config, device)

    loader = build_toptagging_loader(
        args.data_path,
        args.n_jets,
        args.batch_size,
        momentum_float64=bool(cfg.data.get("momentum_float64", True)),
    )

    t_total = time.time()

    # ── Baseline ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Run 0/? — Baseline (no ablation)")
    t0 = time.time()
    labels, scores = run_inference(model, loader, cfg, args.n_jets, device)
    baseline_auc = float(roc_auc_score(labels, scores))
    print(f"  Baseline AUC = {baseline_auc:.5f}  [{time.time()-t0:.0f}s]")

    grade_aucs      = None
    keep_only_aucs  = None
    layer_aucs      = None
    extra_zero_only_results = []
    extra_keep_only_results = []
    scores_to_save = {"labels": labels, "baseline": scores}

    # ── Full-network ablations ────────────────────────────────────────────────
    if args.mode in ("full", "all"):
        # Zero-grade ablation
        print(f"\n{'='*60}")
        print("Full-network ablation (zeroing each intervention at ALL layers)")
        grade_aucs = np.zeros(len(interventions), dtype=np.float32)
        for row, intervention in enumerate(interventions):
            t0 = time.time()
            handles = register_full_network_hooks(model, intervention)
            labels_g, scores_g = run_inference(model, loader, cfg, args.n_jets, device)
            for h in handles:
                h.remove()
            auc_g = float(roc_auc_score(labels_g, scores_g))
            grade_aucs[row] = auc_g
            scores_to_save[f"zero_{intervention['key']}"] = scores_g
            print(
                f"  {intervention['name']:<28}: "
                f"AUC={auc_g:.5f}  "
                f"ΔAUC={baseline_auc - auc_g:+.5f}  "
                f"[{time.time()-t0:.0f}s]"
            )

        # Keep-only ablation (zero all OTHER grades)
        print(f"\n{'='*60}")
        print("Full-network keep-only ablation (keeping ONLY one intervention group)")
        keep_only_aucs = np.zeros(len(interventions), dtype=np.float32)
        for row, intervention in enumerate(interventions):
            t0 = time.time()
            handles = register_keep_only_hooks(model, intervention)
            labels_g, scores_g = run_inference(model, loader, cfg, args.n_jets, device)
            for h in handles:
                h.remove()
            auc_g = float(roc_auc_score(labels_g, scores_g))
            keep_only_aucs[row] = auc_g
            scores_to_save[f"keep_{intervention['key']}"] = scores_g
            print(
                f"  Keep only {intervention['name']:<18}: "
                f"AUC={auc_g:.5f}  "
                f"ΔAUC={baseline_auc - auc_g:+.5f}  "
                f"[{time.time()-t0:.0f}s]"
            )

        if args.include_clean_scalar_side_study:
            print(f"\n{'='*60}")
            print("Appendix scalar side-study (zero G0+G4 and hidden scalar branch)")
            t0 = time.time()
            handles = register_full_network_hooks(model, CLEAN_SCALAR_SIDE_STUDY)
            labels_g, scores_g = run_inference(model, loader, cfg, args.n_jets, device)
            for h in handles:
                h.remove()
            auc_g = float(roc_auc_score(labels_g, scores_g))
            extra_zero_only_results.append((CLEAN_SCALAR_SIDE_STUDY, auc_g))
            scores_to_save[f"zero_{CLEAN_SCALAR_SIDE_STUDY['key']}"] = scores_g
            print(
                f"  {CLEAN_SCALAR_SIDE_STUDY['name']:<28}: "
                f"AUC={auc_g:.5f}  "
                f"ΔAUC={baseline_auc - auc_g:+.5f}  "
                f"[{time.time()-t0:.0f}s]"
            )

        if args.include_clean_keep_only_side_study:
            print(f"\n{'='*60}")
            print("Appendix strict keep-only side-study")
            for intervention in CLEAN_KEEP_ONLY_SIDE_STUDIES:
                t0 = time.time()
                handles = register_keep_only_hooks(model, intervention)
                labels_g, scores_g = run_inference(model, loader, cfg, args.n_jets, device)
                for h in handles:
                    h.remove()
                auc_g = float(roc_auc_score(labels_g, scores_g))
                extra_keep_only_results.append((intervention, auc_g))
                scores_to_save[f"keep_{intervention['key']}"] = scores_g
                print(
                    f"  {intervention['name']:<28}: "
                    f"AUC={auc_g:.5f}  "
                    f"ΔAUC={baseline_auc - auc_g:+.5f}  "
                    f"[{time.time()-t0:.0f}s]"
                )

        if args.save_scores:
            scores_path = output_dir / "grade_decomp_scores.npz"
            np.savez_compressed(scores_path, **scores_to_save)
            print(f"Saved paired-bootstrap score file -> {scores_path}")

        save_kwargs = dict(
            baseline_auc=np.float32(baseline_auc),
            grade_aucs=grade_aucs,
            keep_only_aucs=keep_only_aucs,
            intervention_keys=np.array([item["key"] for item in interventions]),
            intervention_names=np.array([item["name"] for item in interventions]),
            intervention_grade_mask=np.array([item["grade_mask"] for item in interventions], dtype=np.int32),
            grade_names=np.array(GRADE_NAMES),
        )
        if extra_zero_only_results:
            save_kwargs["extra_zero_only_keys"] = np.array([item["key"] for item, _ in extra_zero_only_results])
            save_kwargs["extra_zero_only_names"] = np.array([item["name"] for item, _ in extra_zero_only_results])
            save_kwargs["extra_zero_only_aucs"] = np.array([auc_val for _, auc_val in extra_zero_only_results], dtype=np.float32)
        if extra_keep_only_results:
            save_kwargs["extra_keep_only_keys"] = np.array([item["key"] for item, _ in extra_keep_only_results])
            save_kwargs["extra_keep_only_names"] = np.array([item["name"] for item, _ in extra_keep_only_results])
            save_kwargs["extra_keep_only_aucs"] = np.array([auc_val for _, auc_val in extra_keep_only_results], dtype=np.float32)
        np.savez(output_dir / "grade_decomp_full.npz", **save_kwargs)
        plot_full_network(
            baseline_auc, grade_aucs, keep_only_aucs, interventions,
            args.n_jets, output_dir / "grade_decomp_full.png",
        )

    # ── Layer-resolved grade ablation ─────────────────────────────────────────
    if args.mode in ("layer", "all"):
        print(f"\n{'='*60}")
        print(f"Layer-resolved ablation ({len(interventions)} interventions × 13 layers)")
        layer_aucs = np.zeros((len(interventions), NUM_LAYERS), dtype=np.float32)
        run_idx = 0
        total_runs = len(interventions) * NUM_LAYERS

        for row, intervention in enumerate(interventions):
            for l in range(NUM_LAYERS):
                t0 = time.time()
                handles = register_single_layer_hook(model, intervention, l)
                labels_gl, scores_gl = run_inference(model, loader, cfg, args.n_jets, device)
                for h in handles:
                    h.remove()
                auc_gl = float(roc_auc_score(labels_gl, scores_gl))
                layer_aucs[row, l] = auc_gl
                run_idx += 1

                layer_label = f"L{l}" if l > 0 else "L0(linear_in)"
                print(
                    f"  [{run_idx:3d}/{total_runs}]  "
                    f"{intervention['name']:<28} @ {layer_label:<14}: "
                    f"AUC={auc_gl:.5f}  "
                    f"ΔAUC={baseline_auc - auc_gl:+.5f}  "
                    f"[{time.time()-t0:.0f}s]"
                )

        np.savez(
            output_dir / "grade_decomp_layer.npz",
            baseline_auc=np.float32(baseline_auc),
            layer_aucs=layer_aucs,
            intervention_keys=np.array([item["key"] for item in interventions]),
            intervention_names=np.array([item["name"] for item in interventions]),
            intervention_grade_mask=np.array([item["grade_mask"] for item in interventions], dtype=np.int32),
            grade_names=np.array(GRADE_NAMES),
        )
        plot_layer_resolved(
            baseline_auc, layer_aucs, interventions,
            args.n_jets, output_dir / "grade_decomp_layer.png",
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    write_summary(
        baseline_auc, grade_aucs, keep_only_aucs, layer_aucs, interventions, args.n_jets,
        output_dir / "grade_decomp_summary.txt",
        extra_zero_only_results=extra_zero_only_results,
        extra_keep_only_results=extra_keep_only_results,
    )
    print(f"\nTotal elapsed: {time.time()-t_total:.0f}s")
    print(f"All outputs in {output_dir}")


if __name__ == "__main__":
    main()
