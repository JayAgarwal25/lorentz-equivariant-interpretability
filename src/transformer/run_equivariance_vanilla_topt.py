#!/usr/bin/env python3
"""Per-layer equivariance test for Vanilla-T under random Lorentz transforms and a boost sweep.

The transform is applied to the particle four-momenta of the first 50 particles, and the model's
preprocessing recomputes its inputs. Records the relative change of the mean-pooled hidden state
per layer and of the output logit.

    python src/transformer/run_equivariance_vanilla_topt.py --checkpoint $RUNS/topt_vanilla_transformer/seed1001/models/model_run0.pt \\
        --config $RUNS/topt_vanilla_transformer/seed1001/config_0.yaml --data-path $DATA \\
        --n-jets 200 --n-transforms 5 --n-jets-sweep 50 --seed 1001 \\
        --output-dir results/equivariance_gamma5/vanilla_transformer/seed1001
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
from omegaconf import OmegaConf, open_dict
from hydra.utils import instantiate

sys.path.insert(0, ".")
from experiments.tagging.embedding import embed_tagging_data

N_LAYERS = None # Filled from the loaded model; avoids stale layer-count assumptions.
EPS      = 1e-8


# ── Lorentz transform helpers ─────────────────────────────────────────────────

def boost_matrix(gamma, direction=None, rng=None):
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
    return np.array([
        [gamma,    gb*nx,         gb*ny,         gb*nz        ],
        [gb*nx,    1+gm1*nx*nx,   gm1*nx*ny,     gm1*nx*nz    ],
        [gb*ny,    gm1*nx*ny,     1+gm1*ny*ny,   gm1*ny*nz    ],
        [gb*nz,    gm1*nx*nz,     gm1*ny*nz,     1+gm1*nz*nz  ],
    ], dtype=np.float32)


def rotation_matrix(rng=None):
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
    if rng is None:
        rng = np.random.default_rng()
    gamma   = float(np.exp(rng.uniform(np.log(1.5), np.log(3.0))))
    L_boost = boost_matrix(gamma, rng=rng)
    L_rot   = rotation_matrix(rng)
    return (L_boost @ L_rot).astype(np.float32)


def sample_pure_boost(gamma, rng=None):
    return boost_matrix(gamma, rng=rng)


def apply_lorentz(p4_np, Lambda):
    return (p4_np @ Lambda).astype(np.float32)


def rel_err(a, b, eps=EPS):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + eps))


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(config_path, checkpoint_path, device):
    cfg = OmegaConf.load(config_path)
    with open_dict(cfg):
        cfg.model.net.compile = False
    model = instantiate(cfg.model)
    ckpt  = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    print(f"Loaded model from {checkpoint_path}")
    return model, cfg


# ── Forward pass with hooks ───────────────────────────────────────────────────

def run_model(model, cfg, p4_np, device):
    """
    Run vanilla transformer on a single jet (p4_np: (N, 4) in E,px,py,pz).
    Returns:
      hs_list : list of N_LAYERS arrays, each shape (hidden_dim,) — mean-pooled jet rep
      logit   : float — raw logit for class 1
    """
    N = len(p4_np)
    x_t   = torch.tensor(p4_np, dtype=torch.float32, device=device)
    ptr_t = torch.tensor([0, N], dtype=torch.long, device=device)
    scalars = torch.zeros(N, 0, dtype=torch.float32, device=device)

    embedding = embed_tagging_data(x_t, scalars, ptr_t, cfg.data)
    embedding["num_graphs"] = 1

    n_layers = 1 + len(model.net.blocks)
    hs_raw = [None] * n_layers

    def _make_hook(li):
        def hook(module, inp, out):
            hs_raw[li] = out.detach().cpu()
        return hook

    handles = []
    handles.append(model.net.linear_in.register_forward_hook(_make_hook(0)))
    for li, blk in enumerate(model.net.blocks):
        handles.append(blk.register_forward_hook(_make_hook(li + 1)))

    with torch.no_grad():
        out = model(embedding)

    for h in handles:
        h.remove()

    hs_list = []
    for li in range(n_layers):
        t = hs_raw[li]
        if t is None:
            hs_list.append(None)
        else:
            # Hooks may return either (n_particles, hidden_dim) or
            # (batch=1, n_particles, hidden_dim). Remove the singleton batch
            # dimension before mean-pooling to one jet representation.
            while t.ndim > 2 and t.shape[0] == 1:
                t = t.squeeze(0)
            hs_list.append(t.mean(dim=0).numpy().astype(np.float32))

    # model returns (logits, extras_dict, frames) — logits shape (1, n_classes) or (1, 1)
    logits = out[0]
    logit = float(logits.squeeze()[1].cpu()) if logits.shape[-1] > 1 else float(logits.squeeze().cpu())
    return hs_list, logit


# ── Equivariance test ─────────────────────────────────────────────────────────

def test_equivariance(model, cfg, npz, n_jets, n_transforms, device, rng):
    kinematics = npz["kinematics_test"]  # (N_jets, max_part, 4): E,px,py,pz

    n_layers   = 1 + len(model.net.blocks)
    err_all    = np.zeros((n_jets, n_transforms, n_layers), dtype=np.float32)
    logit_errs = np.zeros((n_jets, n_transforms),           dtype=np.float32)

    transforms = [sample_lorentz_transform(rng) for _ in range(n_transforms)]

    jet_count = 0
    t0 = time.time()

    for ji in range(len(kinematics)):
        if jet_count >= n_jets:
            break

        raw = kinematics[ji]  # (max_part, 4)
        E_a, px_a, py_a, pz_a = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]
        is_real = (E_a != 0) | (px_a != 0) | (py_a != 0) | (pz_a != 0)
        if is_real.sum() < 3:
            continue

        idx = np.where(is_real)[0][:50]
        p4  = np.column_stack([E_a[idx], px_a[idx], py_a[idx], pz_a[idx]]).astype(np.float32)

        hs_orig, logit_orig = run_model(model, cfg, p4, device)

        for ti, Lambda in enumerate(transforms):
            p4_b = apply_lorentz(p4, Lambda)
            hs_boost, logit_boost = run_model(model, cfg, p4_b, device)

            logit_errs[jet_count, ti] = abs(logit_boost - logit_orig) / (abs(logit_orig) + EPS)
            for li in range(n_layers):
                if hs_orig[li] is not None and hs_boost[li] is not None:
                    err_all[jet_count, ti, li] = rel_err(hs_boost[li], hs_orig[li])

        jet_count += 1
        if jet_count % 50 == 0:
            print(f"  Vanilla: {jet_count}/{n_jets} jets  [{time.time()-t0:.0f}s]")

    return err_all[:jet_count], logit_errs[:jet_count]


def test_boost_sweep(model, cfg, npz, gammas, n_jets, n_transforms, device, rng):
    kinematics = npz["kinematics_test"]
    n_g = len(gammas)
    n_layers = 1 + len(model.net.blocks)

    sweep_err   = np.zeros((n_g, n_jets, n_transforms), dtype=np.float32)
    sweep_logit = np.zeros((n_g, n_jets, n_transforms), dtype=np.float32)

    jets = []
    for ji in range(len(kinematics)):
        if len(jets) >= n_jets:
            break
        raw = kinematics[ji]
        E_a, px_a, py_a, pz_a = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]
        is_real = (E_a != 0) | (px_a != 0) | (py_a != 0) | (pz_a != 0)
        if is_real.sum() < 3:
            continue
        idx = np.where(is_real)[0][:50]
        jets.append(np.column_stack([raw[idx, 0], raw[idx, 1], raw[idx, 2], raw[idx, 3]]).astype(np.float32))

    t0 = time.time()
    for gi, gamma in enumerate(gammas):
        transforms = [sample_pure_boost(gamma, rng) for _ in range(n_transforms)]
        for ji, p4 in enumerate(jets):
            hs_orig, logit_orig = run_model(model, cfg, p4, device)
            for ti, Lambda in enumerate(transforms):
                p4_b = apply_lorentz(p4, Lambda)
                hs_boost, logit_boost = run_model(model, cfg, p4_b, device)
                sweep_logit[gi, ji, ti] = abs(logit_boost - logit_orig) / (abs(logit_orig) + EPS)
                layer_errs = [rel_err(hs_boost[li], hs_orig[li])
                              for li in range(n_layers)
                              if hs_orig[li] is not None and hs_boost[li] is not None]
                sweep_err[gi, ji, ti] = float(np.mean(layer_errs))

        print(f"  γ={gamma:5.1f}: hidden_err={sweep_err[gi].mean():.3f}  "
              f"logit_err={sweep_logit[gi].mean():.3f}  [{time.time()-t0:.0f}s]")

    return sweep_err, sweep_logit


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_equivariance(err_all, n_jets, n_tr, lgatr_ref_lo, lgatr_ref_hi, save_path):
    n_j, n_t, n_layers = err_all.shape
    flat  = err_all.reshape(-1, n_layers)
    mean  = flat.mean(0)
    std   = flat.std(0)
    layers       = np.arange(n_layers)
    layer_labels = ["L0\n(linear_in)"] + [f"L{i}" for i in range(1, n_layers)]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.semilogy(layers, mean, "r-o", linewidth=2, markersize=5,
                label="Vanilla Transformer  ||h(Λx)−h(x)||/||h(x)||")
    ax.fill_between(layers, np.maximum(mean - std, 1e-9), mean + std,
                    alpha=0.2, color="red")
    ax.axhline(lgatr_ref_hi, color="blue",           linestyle="--", linewidth=1.5,
               label=f"LGATr h_s  (max ~{lgatr_ref_hi:.1e})")
    ax.axhline(lgatr_ref_lo, color="cornflowerblue", linestyle=":",  linewidth=1.2,
               label=f"LGATr h_s  (min ~{lgatr_ref_lo:.1e})")
    ax.axhline(1.2e-7, color="gray",                 linestyle=":",  linewidth=1.0,
               label="float32 machine ε")
    ax.set_xticks(layers)
    ax.set_xticklabels(layer_labels, fontsize=9)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Relative change in jet rep  (log scale)", fontsize=12)
    ax.set_title(
        f"Vanilla Transformer — Per-layer Lorentz Invariance Error\n"
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
              label="Vanilla Transformer hidden state change")
    ax.loglog(gammas, mean_logit, "g-^", linewidth=2, markersize=7,
              label="Vanilla Transformer logit change")
    ax.axhline(lgatr_logit_ref, color="blue", linestyle="--", linewidth=1.5,
               label=f"LGATr logit invariance (ref {lgatr_logit_ref:.1e})")
    ax.set_xlabel("Lorentz boost factor γ", fontsize=12)
    ax.set_ylabel("Mean relative change", fontsize=12)
    ax.set_title("Vanilla Transformer invariance vs boost strength", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved boost sweep → {save_path}")


def write_summary(err_all, logit_errs, gammas, sweep_err, sweep_logit,
                  n_jets, n_tr, lgatr_logit_ref, save_path):
    n_layers = err_all.shape[-1]
    layer_labels = ["L0(linear_in)"] + [f"L{i}" for i in range(1, n_layers)]
    flat     = err_all.reshape(-1, n_layers)
    mean_err = flat.mean(0)

    lines = [
        "Per-layer Lorentz Invariance Test  —  Vanilla Transformer (TopTagging)",
        f"N_jets={n_jets}  N_transforms={n_tr}",
        "=" * 70,
        "",
        "Metric: ||h_l(Λx) - h_l(x)||_F / ||h_l(x)||_F  (mean-pooled jet rep)",
        "Vanilla Transformer is NOT Lorentz-invariant — large errors are expected.",
        "",
        f"{'Layer':<20}  {'Mean hidden state change':>24}",
    ]
    for li in range(n_layers):
        lines.append(f"  {layer_labels[li]:<18}  {mean_err[li]:>24.4f}")

    lines += [
        "",
        "Output logit change: |logit(Λx) - logit(x)| / |logit(x)|",
        f"  Vanilla mean: {logit_errs.mean():.4f}",
        f"  LGATr ref:    {lgatr_logit_ref:.2e}",
        f"  Ratio (Vanilla/LGATr): {logit_errs.mean() / lgatr_logit_ref:.1f}x",
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
    ap.add_argument("--config",
        required=True)
    ap.add_argument("--checkpoint",
        required=True)
    ap.add_argument("--data-path",    default="data/toptagging_full.npz")
    ap.add_argument("--output-dir",   default="results/equivariance_vanilla_topt")
    ap.add_argument("--n-jets",       type=int, default=200)
    ap.add_argument("--n-transforms", type=int, default=5)
    ap.add_argument("--n-jets-sweep", type=int, default=50)
    ap.add_argument("--gammas", type=float, nargs="+",
                    default=[1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0],
                    help="Lorentz boost factors for the pure-boost sweep")
    ap.add_argument("--seed",         type=int, default=42)
    args = ap.parse_args()

    GAMMAS          = args.gammas
    LGATR_LOGIT_REF = 5.61e-3
    LGATR_REF_LO    = 1e-4
    LGATR_REF_HI    = 5e-3

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng    = np.random.default_rng(args.seed)
    print(f"Device: {device}")
    print(f"n_jets={args.n_jets}  n_transforms={args.n_transforms}  seed={args.seed}")

    model, cfg = load_model(args.config, args.checkpoint, device)

    print(f"\nLoading {args.data_path} ...")
    npz = np.load(args.data_path)

    t_total = time.time()

    print(f"\n{'='*60}")
    print(f"Per-layer invariance test ({args.n_jets} jets × {args.n_transforms} transforms)...")
    err_all, logit_errs = test_equivariance(
        model, cfg, npz, args.n_jets, args.n_transforms, device, rng
    )
    print(f"  Mean logit change:  {logit_errs.mean():.4f}")
    print(f"  LGATr logit ref:    {LGATR_LOGIT_REF:.2e}")

    print(f"\n{'='*60}")
    print(f"Boost sweep γ={GAMMAS} ({args.n_jets_sweep} jets each)...")
    sweep_err, sweep_logit = test_boost_sweep(
        model, cfg, npz, GAMMAS, args.n_jets_sweep, args.n_transforms, device, rng
    )

    np.savez(
        output_dir / "equivariance_vanilla_topt.npz",
        err_all=err_all, logit_errs=logit_errs,
        gammas=np.array(GAMMAS),
        sweep_err=sweep_err, sweep_logit=sweep_logit,
    )
    print(f"Saved data → {output_dir}/equivariance_vanilla_topt.npz")

    plot_equivariance(err_all, len(err_all), args.n_transforms,
                      LGATR_REF_LO, LGATR_REF_HI,
                      save_path=output_dir / "equivariance_vanilla_topt.png")
    plot_boost_sweep(GAMMAS, sweep_err, sweep_logit, LGATR_LOGIT_REF,
                     save_path=output_dir / "equivariance_boost_sweep_vanilla_topt.png")
    write_summary(err_all, logit_errs, GAMMAS, sweep_err, sweep_logit,
                  len(err_all), args.n_transforms, LGATR_LOGIT_REF,
                  output_dir / "equivariance_summary_vanilla_topt.txt")

    print(f"\nTotal elapsed: {time.time()-t_total:.0f}s")
    print(f"All outputs in {output_dir}")


if __name__ == "__main__":
    main()
