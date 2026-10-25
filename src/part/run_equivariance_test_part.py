#!/usr/bin/env python3
"""
Per-layer Lorentz invariance test for ParT (ParticleTransformer) on TopTagging.

ParT is NOT Lorentz-equivariant by design — it uses coordinates (eta, phi, pt)
that transform non-trivially under general Lorentz boosts, and its attention
mechanism has no equivariance constraints.

This script measures how much the hidden representations and output logit change
when we apply a random proper orthochronous Lorentz transformation to the input
jet 4-momenta (the features are recomputed from the boosted 4-momenta each time).

Metric (per layer):
  err[l] = ||h_l(Λx) - h_l(x)||_F / (||h_l(x)||_F + ε)

Expected:
  - err >> 0 for all layers (ParT is not Lorentz-invariant)
  - Logit change >> LGATr reference (5.61e-3)

No clifford/GA library needed — Lorentz transforms implemented in pure numpy.

Usage:
  cd path/to/particle_transformer
  CKPT=training/TopLandscape/ParT/20260418-194812_example_ParticleTransformer_ranger_lr0.001_batch512/net_best_epoch_state.pt
  CUDA_VISIBLE_DEVICES=1 python3.10 run_equivariance_test_part.py \\
      --checkpoint $CKPT \\
      --data-path data/toptagging_full.npz \\
      --output-dir results/equivariance_part/ \\
      --n-jets 200 --n-transforms 5 \\
      > logs/equivariance_part.log 2>&1 &
"""

import argparse
import sys
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, ".")
from networks.example_ParticleTransformer import ParticleTransformerWrapper

N_LAYERS  = 9    # L0 = after embed, L1..L8 = after each of 8 main blocks
N_BLOCKS  = 8
NUM_HEADS = 8
EPS       = 1e-8

# ── Lorentz transform helpers (pure numpy, no clifford) ───────────────────────

def boost_matrix(gamma, direction=None, rng=None):
    """
    4×4 Lorentz boost matrix, row-vector convention: p' = p @ Λ
    p = (E, px, py, pz) row vector.

    For boost in direction n̂ with Lorentz factor γ:
      E'   = γ E + γβ (n·p_3)
      p'_i = γβ n_i E + p_i + (γ-1) n_i (n·p_3)
    which is symmetric in n, so Λ = Λ^T.
    """
    if rng is None:
        rng = np.random.default_rng()
    if direction is None:
        n = rng.standard_normal(3)
        n = n / np.linalg.norm(n)
    else:
        n = np.asarray(direction, dtype=float)
        n = n / np.linalg.norm(n)
    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    gb   = gamma * beta
    gm1  = gamma - 1.0
    nx, ny, nz = n
    L = np.array([
        [gamma,    gb*nx,            gb*ny,            gb*nz           ],
        [gb*nx,    1 + gm1*nx*nx,    gm1*nx*ny,        gm1*nx*nz       ],
        [gb*ny,    gm1*nx*ny,        1 + gm1*ny*ny,    gm1*ny*nz       ],
        [gb*nz,    gm1*nx*nz,        gm1*ny*nz,        1 + gm1*nz*nz   ],
    ], dtype=np.float32)
    return L


def rotation_matrix(rng=None):
    """
    Haar-random SO(3) rotation embedded as 4×4 Lorentz matrix.
    E is untouched; spatial 3-momentum is rotated.
    """
    if rng is None:
        rng = np.random.default_rng()
    M = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(M)
    Q = Q * np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    L = np.eye(4, dtype=np.float32)
    L[1:4, 1:4] = Q.astype(np.float32)
    return L


def sample_lorentz_transform(rng=None):
    """
    Random proper orthochronous Lorentz transform: boost then rotation.
    γ sampled log-uniformly in [1.5, 3.0] to avoid near-identity transforms.
    Returns Lambda_4x4 (row-vector convention, p' = p @ Lambda).
    """
    if rng is None:
        rng = np.random.default_rng()
    gamma  = float(np.exp(rng.uniform(np.log(1.5), np.log(3.0))))
    L_boost = boost_matrix(gamma, rng=rng)
    L_rot   = rotation_matrix(rng)
    return (L_boost @ L_rot).astype(np.float32)


def sample_pure_boost(gamma, rng=None):
    """Pure boost with specified γ in a random direction."""
    return boost_matrix(gamma, rng=rng)


def apply_lorentz_to_p4(p4_np, Lambda):
    """p4_np: (N, 4) in (E, px, py, pz); returns (N, 4) after transform."""
    return (p4_np @ Lambda).astype(np.float32)


def rel_err(a, b, eps=EPS):
    """Relative Frobenius norm error: ||a-b||_F / (||b||_F + eps)."""
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + eps))


# ── Model loading ─────────────────────────────────────────────────────────────

def _normalise_checkpoint_state(state):
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    converted = {}
    for key, value in state.items():
        if key.startswith("trafo_fourmomenta.") or ".attention.transform." in key:
            continue
        if key.startswith("net."):
            key = "mod." + key[4:]
        key = key.replace(".attn.in_proj.weight", ".attn.in_proj_weight")
        key = key.replace(".attn.in_proj.bias", ".attn.in_proj_bias")
        converted[key] = value
    return converted


def load_model(checkpoint, device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = _normalise_checkpoint_state(state)
    num_classes = int(state["mod.fc.0.weight"].shape[0])
    cfg = dict(
        input_dim=7,
        num_classes=num_classes,
        pair_input_dim=4,
        use_pre_activation_pair=False,
        embed_dims=[128, 512, 128],
        pair_embed_dims=[64, 64, 64],
        num_heads=NUM_HEADS,
        num_layers=N_BLOCKS,
        num_cls_layers=2,
        block_params=None,
        cls_block_params={"dropout": 0, "attn_dropout": 0, "activation_dropout": 0},
        fc_params=[],
        activation="gelu",
        trim=True,
        for_inference=False,
    )
    model = ParticleTransformerWrapper(**cfg)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"Loaded model ({sum(p.numel() for p in model.parameters()):,} params, num_classes={num_classes})")
    return model


# ── Data preprocessing ────────────────────────────────────────────────────────

def preprocess_jet(E, px, py, pz):
    """
    Build ParT input tensors from raw 4-momenta arrays (E, px, py, pz).
    Features and coordinates are recomputed fresh — this is what makes the test
    meaningful: applying Λ changes the feature values, exposing non-invariance.
    Returns: features_t (1,7,P), points_t (1,2,P), vectors_t (1,4,P), mask_t (1,1,P)
    """
    pt  = np.sqrt(px**2 + py**2)
    eta = np.arcsinh(pz / (pt + 1e-10))
    phi = np.arctan2(py, px)

    jet_px  = px.sum();  jet_py  = py.sum()
    jet_pt  = np.sqrt(jet_px**2 + jet_py**2)
    jet_eta = np.arcsinh(pz.sum() / (jet_pt + 1e-10))
    jet_phi = np.arctan2(jet_py, jet_px)
    jet_E   = E.sum()

    deta = eta - jet_eta
    dphi = ((phi - jet_phi) + np.pi) % (2 * np.pi) - np.pi
    dR   = np.sqrt(deta**2 + dphi**2)

    log_pt     = (np.log(pt + 1e-10) - 1.7) * 0.7
    log_e      = (np.log(E  + 1e-10) - 2.0) * 0.7
    log_pt_rel = (np.log(pt / (jet_pt + 1e-10) + 1e-10) - (-4.7)) * 0.7
    log_e_rel  = (np.log(E  / (jet_E  + 1e-10) + 1e-10) - (-4.7)) * 0.7
    dR_feat    = (dR - 0.2) * 4.0

    feats  = np.clip(np.stack([log_pt, log_e, log_pt_rel, log_e_rel, dR_feat, deta, dphi]), -5, 5)
    points = np.stack([deta, dphi])
    vecs   = np.stack([px, py, pz, E])

    features_t = torch.tensor(feats[np.newaxis],  dtype=torch.float32)
    points_t   = torch.tensor(points[np.newaxis], dtype=torch.float32)
    vectors_t  = torch.tensor(vecs[np.newaxis],   dtype=torch.float32)
    mask_t     = torch.ones(1, 1, len(E), dtype=torch.bool)
    return features_t, points_t, vectors_t, mask_t


# ── Forward pass with hooks ───────────────────────────────────────────────────

def run_part(model, features_t, points_t, vectors_t, mask_t, device):
    """
    Forward pass with hooks on embed (L0) and all 8 main blocks (L1-L8).
    Returns:
      hs_list : list of N_LAYERS arrays, each (P, 128) float32
      logit   : float — raw score for class 1 (top/signal)
    """
    hs_raw = [None] * N_LAYERS

    def _make_hook(li):
        def hook(module, inp, out):
            # out shape: (P, 1, embed_dim) — seq_len first, batch=1 second
            hs_raw[li] = out.detach().cpu()
        return hook

    hooks = []
    hooks.append(model.mod.embed.register_forward_hook(_make_hook(0)))
    for bi, blk in enumerate(model.mod.blocks):
        hooks.append(blk.register_forward_hook(_make_hook(bi + 1)))

    with torch.no_grad():
        out = model(
            points_t.to(device), features_t.to(device),
            lorentz_vectors=vectors_t.to(device),
            mask=mask_t.to(device),
        )
    for h in hooks:
        h.remove()

    hs_list = []
    for li in range(N_LAYERS):
        t = hs_raw[li]
        if t is None:
            hs_list.append(None)
        else:
            # (P, 1, 128) → (P, 128)
            hs_list.append(t.squeeze(1).numpy().astype(np.float32))

    logit = float(out.squeeze().cpu()) if out.shape[-1] == 1 else float(out[0, 1].cpu())
    return hs_list, logit


# ── Main equivariance test ─────────────────────────────────────────────────────

def test_equivariance(model, data, n_jets, n_transforms, device, rng):
    """
    For each jet: run original 4-momenta and n_transforms Lorentz-boosted versions.
    Returns:
      err_all    : (n_jets, n_transforms, N_LAYERS)
      logit_errs : (n_jets, n_transforms)
    """
    kinematics = data["kinematics_test"]
    labels     = data["labels_test"]

    err_all    = np.zeros((n_jets, n_transforms, N_LAYERS), dtype=np.float32)
    logit_errs = np.zeros((n_jets, n_transforms), dtype=np.float32)

    transforms = [sample_lorentz_transform(rng) for _ in range(n_transforms)]

    jet_count = 0
    t0 = time.time()
    for ji in range(len(labels)):
        if jet_count >= n_jets:
            break
        raw  = kinematics[ji]
        E_a, px_a, py_a, pz_a = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]
        is_real = (E_a != 0) | (px_a != 0) | (py_a != 0) | (pz_a != 0)
        if is_real.sum() < 3:
            continue

        idx = np.where(is_real)[0][:50]
        p4  = np.column_stack([E_a[idx], px_a[idx], py_a[idx], pz_a[idx]]).astype(np.float32)

        feat_t, pts_t, vec_t, mask_t = preprocess_jet(p4[:, 0], p4[:, 1], p4[:, 2], p4[:, 3])
        hs_orig, logit_orig = run_part(model, feat_t, pts_t, vec_t, mask_t, device)

        for ti, Lambda in enumerate(transforms):
            p4_b = apply_lorentz_to_p4(p4, Lambda)
            feat_b, pts_b, vec_b, mask_b = preprocess_jet(p4_b[:, 0], p4_b[:, 1], p4_b[:, 2], p4_b[:, 3])
            hs_boost, logit_boost = run_part(model, feat_b, pts_b, vec_b, mask_b, device)

            logit_errs[jet_count, ti] = abs(logit_boost - logit_orig) / (abs(logit_orig) + EPS)
            for li in range(N_LAYERS):
                if hs_orig[li] is not None and hs_boost[li] is not None:
                    err_all[jet_count, ti, li] = rel_err(hs_boost[li], hs_orig[li])

        jet_count += 1
        if jet_count % 50 == 0:
            print(f"  ParT: {jet_count}/{n_jets} jets  [{time.time()-t0:.0f}s]")

    return err_all[:jet_count], logit_errs[:jet_count]


def test_boost_sweep(model, data, gammas, n_jets, n_transforms, device, rng):
    """Sweep pure boosts over specified γ values."""
    kinematics = data["kinematics_test"]
    labels     = data["labels_test"]
    n_g = len(gammas)

    sweep_err   = np.zeros((n_g, n_jets, n_transforms), dtype=np.float32)
    sweep_logit = np.zeros((n_g, n_jets, n_transforms), dtype=np.float32)

    jets = []
    for ji in range(len(labels)):
        if len(jets) >= n_jets:
            break
        raw  = kinematics[ji]
        E_a, px_a, py_a, pz_a = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]
        is_real = (E_a != 0) | (px_a != 0) | (py_a != 0) | (pz_a != 0)
        if is_real.sum() < 3:
            continue
        idx = np.where(is_real)[0][:50]
        p4  = np.column_stack([raw[idx, 0], raw[idx, 1], raw[idx, 2], raw[idx, 3]]).astype(np.float32)
        jets.append(p4)

    t0 = time.time()
    for gi, gamma in enumerate(gammas):
        transforms = [sample_pure_boost(gamma, rng) for _ in range(n_transforms)]
        for ji, p4 in enumerate(jets):
            feat_t, pts_t, vec_t, mask_t = preprocess_jet(p4[:, 0], p4[:, 1], p4[:, 2], p4[:, 3])
            hs_orig, logit_orig = run_part(model, feat_t, pts_t, vec_t, mask_t, device)

            for ti, Lambda in enumerate(transforms):
                p4_b = apply_lorentz_to_p4(p4, Lambda)
                feat_b, pts_b, vec_b, mask_b = preprocess_jet(p4_b[:, 0], p4_b[:, 1], p4_b[:, 2], p4_b[:, 3])
                hs_boost, logit_boost = run_part(model, feat_b, pts_b, vec_b, mask_b, device)

                sweep_logit[gi, ji, ti] = abs(logit_boost - logit_orig) / (abs(logit_orig) + EPS)
                layer_errs = [rel_err(hs_boost[li], hs_orig[li])
                              for li in range(N_LAYERS)
                              if hs_orig[li] is not None and hs_boost[li] is not None]
                sweep_err[gi, ji, ti] = float(np.mean(layer_errs))

        print(f"  γ={gamma:5.1f}: hidden_err={sweep_err[gi].mean():.3f}  "
              f"logit_err={sweep_logit[gi].mean():.3f}  [{time.time()-t0:.0f}s]")

    return sweep_err, sweep_logit


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_equivariance(err_all, n_jets, n_tr, save_path):
    n_j, n_t, n_layers = err_all.shape
    flat = err_all.reshape(-1, n_layers)
    mean = flat.mean(0)
    std  = flat.std(0)
    layers       = np.arange(n_layers)
    layer_labels = ["L0\n(embed)"] + [f"L{i}" for i in range(1, n_layers)]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.semilogy(layers, mean, "r-o", linewidth=2, markersize=5,
                label="ParT  ||h(Λx) − h(x)|| / ||h(x)||")
    ax.fill_between(layers,
                    np.maximum(mean - std, 1e-9),
                    mean + std, alpha=0.2, color="red")
    ax.axhline(5e-3, color="blue",          linestyle="--", linewidth=1.5,
               label="LGATr h_s  (max ~5e-3)")
    ax.axhline(1e-4, color="cornflowerblue", linestyle=":",  linewidth=1.2,
               label="LGATr h_s  (min ~1e-4)")
    ax.axhline(1.2e-7, color="gray",        linestyle=":",  linewidth=1.0,
               label="float32 machine ε")
    ax.set_xticks(layers)
    ax.set_xticklabels(layer_labels, fontsize=9)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Relative change in hidden state  (log scale)", fontsize=12)
    ax.set_title(
        f"ParT — Per-layer Lorentz Invariance Error\n"
        f"({n_jets} jets × {n_tr} transforms, γ ∈ [1.5, 3.0])",
        fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")


def plot_boost_sweep(gammas, sweep_err, sweep_logit, lgatr_logit_ref, save_path):
    mean_err   = sweep_err.reshape(len(gammas), -1).mean(1)
    mean_logit = sweep_logit.reshape(len(gammas), -1).mean(1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(gammas, mean_err,   "r-o", linewidth=2, markersize=7,
              label="ParT hidden state change")
    ax.loglog(gammas, mean_logit, "g-^", linewidth=2, markersize=7,
              label="ParT logit change")
    ax.axhline(lgatr_logit_ref, color="blue", linestyle="--", linewidth=1.5,
               label=f"LGATr logit invariance (ref {lgatr_logit_ref:.1e})")
    ax.set_xlabel("Lorentz boost factor γ", fontsize=12)
    ax.set_ylabel("Mean relative change", fontsize=12)
    ax.set_title("ParT invariance vs boost strength  (contrast with LGATr)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved boost sweep → {save_path}")


def write_summary(err_all, logit_errs, gammas, sweep_err, sweep_logit,
                  n_jets, n_tr, lgatr_logit_ref, save_path):
    layer_labels = ["L0(embed)"] + [f"L{i}" for i in range(1, N_LAYERS)]
    flat     = err_all.reshape(-1, N_LAYERS)
    mean_err = flat.mean(0)

    lines = [
        "Per-layer Lorentz Invariance Test  —  ParT (ParticleTransformer)",
        f"N_jets={n_jets}  N_transforms={n_tr}",
        "=" * 70,
        "",
        "Metric: ||h_l(Λx) - h_l(x)||_F / ||h_l(x)||_F",
        "ParT is NOT Lorentz-invariant — large errors are expected and meaningful.",
        "",
        f"{'Layer':<18}  {'Mean hidden state change':>24}",
    ]
    for li in range(N_LAYERS):
        lines.append(f"  {layer_labels[li]:<16}  {mean_err[li]:>24.4f}")

    lines += [
        "",
        "Output logit change: |logit(Λx) - logit(x)| / |logit(x)|",
        f"  ParT  mean: {logit_errs.mean():.4f}",
        f"  LGATr ref:  {lgatr_logit_ref:.2e}  (from prior equivariance test)",
        f"  Ratio (ParT/LGATr): {logit_errs.mean() / lgatr_logit_ref:.1f}x",
        "",
        "Boost sweep — pure boosts with increasing γ (mean over all layers + jets):",
        f"  {'γ':>6}  {'hidden change':>14}  {'logit change':>13}",
    ]
    for gi, gamma in enumerate(gammas):
        lines.append(
            f"  {gamma:>6.1f}  {sweep_err[gi].mean():>14.4f}  {sweep_logit[gi].mean():>13.4f}"
        )

    text = "\n".join(lines)
    save_path.write_text(text)
    print(f"\nSaved summary → {save_path}")
    print(text)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ckpt_default = (
        "training/TopLandscape/ParT/"
        "20260418-194812_example_ParticleTransformer_ranger_lr0.001_batch512/"
        "net_best_epoch_state.pt"
    )
    ap.add_argument("--checkpoint",   default=ckpt_default)
    ap.add_argument("--data-path",    default="data/toptagging_full.npz")
    ap.add_argument("--output-dir",   default="results/equivariance_part")
    ap.add_argument("--n-jets",       type=int, default=200)
    ap.add_argument("--n-transforms", type=int, default=5)
    ap.add_argument("--n-jets-sweep", type=int, default=50)
    ap.add_argument("--gammas",       type=float, nargs="+",
                    default=[1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0])
    ap.add_argument("--seed",         type=int, default=42)
    args = ap.parse_args()

    GAMMAS         = args.gammas
    LGATR_LOGIT_REF = 5.61e-3

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng    = np.random.default_rng(args.seed)
    print(f"Device: {device}")
    print(f"n_jets={args.n_jets}  n_transforms={args.n_transforms}  seed={args.seed}")

    model = load_model(args.checkpoint, device)

    print(f"\nLoading {args.data_path} ...")
    npz  = np.load(args.data_path)
    data = {"kinematics_test": npz["kinematics_test"],
            "labels_test":     npz["labels_test"]}

    t_total = time.time()

    print(f"\n{'='*60}")
    print(f"Per-layer invariance test ({args.n_jets} jets × {args.n_transforms} transforms)...")
    err_all, logit_errs = test_equivariance(
        model, data, args.n_jets, args.n_transforms, device, rng
    )
    print(f"  Mean logit change:  {logit_errs.mean():.4f}")
    print(f"  LGATr logit ref:    {LGATR_LOGIT_REF:.2e}")

    print(f"\n{'='*60}")
    print(f"Boost sweep γ={GAMMAS} ({args.n_jets_sweep} jets each)...")
    sweep_err, sweep_logit = test_boost_sweep(
        model, data, GAMMAS, args.n_jets_sweep, args.n_transforms, device, rng
    )

    np.savez(
        output_dir / "equivariance_part.npz",
        err_all=err_all, logit_errs=logit_errs,
        gammas=np.array(GAMMAS),
        sweep_err=sweep_err, sweep_logit=sweep_logit,
    )
    print(f"Saved data → {output_dir}/equivariance_part.npz")

    plot_equivariance(err_all, len(err_all), args.n_transforms,
                      save_path=output_dir / "equivariance_part.png")
    plot_boost_sweep(GAMMAS, sweep_err, sweep_logit, LGATR_LOGIT_REF,
                     save_path=output_dir / "equivariance_boost_sweep_part.png")
    write_summary(err_all, logit_errs, GAMMAS, sweep_err, sweep_logit,
                  len(err_all), args.n_transforms, LGATR_LOGIT_REF,
                  output_dir / "equivariance_summary_part.txt")

    print(f"\nTotal elapsed: {time.time()-t_total:.0f}s")
    print(f"All outputs in {output_dir}")


if __name__ == "__main__":
    main()
