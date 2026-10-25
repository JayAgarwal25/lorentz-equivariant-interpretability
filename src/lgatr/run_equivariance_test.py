#!/usr/bin/env python3
"""Per-layer equivariance test for L-GATr under random Lorentz transforms and a boost sweep.

The transform is applied to the embedded multivectors of every token, particles and spurions;
the scalar inputs are left unchanged. Records the relative change of h_s, of h_mv against
rho(Lambda) h_mv, of h_mv against no transform (naive), and of the output logit.

    cd path/to/lorentz-gatr    # its experiments package provides embed_tagging_data_into_ga
    python path/to/src/lgatr/run_equivariance_test.py --lgatr-checkpoint $RUNS/topt_lgatr/seed1001/models/model_run0.pt \\
        --lgatr-config $RUNS/topt_lgatr/seed1001/config_0.yaml --data-path $DATA \\
        --n-jets 200 --n-transforms 5 --n-jets-sweep 50 --seed 1001 \\
        --output-dir results/equivariance_gamma5/lgatr/seed1001
"""

import sys
import argparse
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from omegaconf import OmegaConf
from torch_geometric.loader import DataLoader

sys.path.insert(0, ".")

from experiments.tagging.dataset import TopTaggingDataset
from experiments.tagging.embedding import embed_tagging_data_into_ga

NUM_LAYERS = 13
EPS = 1e-8

# ── Lorentz transform helpers ─────────────────────────────────────────────────

def sample_lorentz_transform(rng=None):
    """
    Sample a random proper orthochronous Lorentz transformation.

    Returns:
      u        : clifford Pin(1,3) element  (Spin element, even grade)
      rho_16   : (16,16) float32 numpy array — GA representation matrix
                 rho_16[i,j] = j-th component of ρ(Λ)·e_i
                 So:  ρ(Λ)·v  =  (v @ rho_16)   (treating v as row vector)
      Lambda_4x4 : (4,4) float32 numpy array — Lorentz matrix in (E,px,py,pz) basis
                   p' = p @ Lambda_4x4.T  (acts on row vectors)
    """
    from gatr.utils.clifford import sample_pin_multivector, sandwich, np_to_mv

    if rng is None:
        rng = np.random.default_rng()

    u = sample_pin_multivector(spin=True, rng=rng)

    # Compute 16×16 GA representation matrix:
    # Apply sandwich product to each of the 16 standard basis multivectors.
    # Standard basis: e_i has value[i]=1, all others 0.
    rho_16 = np.zeros((16, 16), dtype=np.float32)
    for i in range(16):
        basis_vec = np.zeros(16)
        basis_vec[i] = 1.0
        basis_mv = np_to_mv(basis_vec)
        transformed = sandwich(u, basis_mv)
        rho_16[i] = transformed.value.astype(np.float32)

    # 4×4 Lorentz matrix: grade-1 block of rho_16
    # embed_vector puts 4-momentum p at indices 1-4 (grade-1 components)
    # Lambda_4x4[i,j] = rho_16[1+i, 1+j]  for i,j in 0..3
    Lambda_4x4 = rho_16[1:5, 1:5].copy()

    return u, rho_16, Lambda_4x4


def _rho_from_u(u):
    """Compute 16×16 GA representation matrix from a clifford spinor u."""
    from gatr.utils.clifford import sandwich, np_to_mv
    rho_16 = np.zeros((16, 16), dtype=np.float32)
    for i in range(16):
        basis_vec = np.zeros(16)
        basis_vec[i] = 1.0
        rho_16[i] = sandwich(u, np_to_mv(basis_vec)).value.astype(np.float32)
    return rho_16


def sample_pure_boost(gamma, rng=None):
    """
    Sample a pure Lorentz boost with Lorentz factor gamma in a random direction.

    Boost spinor in Cl(1,3): u = cosh(η/2) + sinh(η/2)*(nx·e12 + ny·e13 + nz·e14)
    where e12, e13, e14 are the boost bivectors (square to +1).
    """
    from gatr.utils.clifford import LAYOUT, np_to_mv

    if rng is None:
        rng = np.random.default_rng()

    eta = float(np.arccosh(float(gamma)))
    n = rng.standard_normal(3)
    n = n / np.linalg.norm(n)

    ch, sh = np.cosh(eta / 2), np.sinh(eta / 2)

    blade_list = LAYOUT.bladeTupList
    idx_e12 = blade_list.index((1, 2))
    idx_e13 = blade_list.index((1, 3))
    idx_e14 = blade_list.index((1, 4))

    u_val = np.zeros(16)
    u_val[0]       = ch
    u_val[idx_e12] = sh * n[0]
    u_val[idx_e13] = sh * n[1]
    u_val[idx_e14] = sh * n[2]

    u = np_to_mv(u_val)
    rho_16 = _rho_from_u(u)
    return u, rho_16, rho_16[1:5, 1:5].copy()


def apply_lorentz_to_p4(p4_np, Lambda_4x4):
    """
    Apply Lorentz transform to array of 4-momenta.
    p4_np : (N, 4) in (E, px, py, pz) convention.
    rho_16[i,j] = j-th component of ρ(Λ)·e_i, so for row vector p:
      (ρ(Λ)·p)_j = Σ_i p_i * rho_16[i,j]  =>  p_new = p @ Lambda_4x4  (no .T)
    """
    return (p4_np @ Lambda_4x4).astype(np.float32)


def apply_rho_to_hmv(h_mv_np, rho_16):
    """
    Apply the GA representation matrix to h_mv.
    h_mv_np : (..., n_ch, 16) numpy array
    Returns (..., n_ch, 16)
    Computes: h_mv @ rho_16 along the last dim.
    """
    return (h_mv_np @ rho_16).astype(np.float32)


def apply_lambda_to_hv(h_v_np, Lambda_4x4):
    """
    Apply Lorentz matrix to h_v.
    h_v_np : (..., n_ch, 4) numpy array
    Returns (..., n_ch, 4)
    Computes: h_v @ Lambda_4x4.T along last dim.
    """
    return (h_v_np @ Lambda_4x4.T).astype(np.float32)


def rel_err(a, b, eps=EPS):
    """Relative Frobenius norm error: ||a-b|| / (||b|| + eps)."""
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + eps))


# ── LGATr (full) equivariance ─────────────────────────────────────────────────

def load_lgatr(checkpoint_path, config_path, device):
    from hydra.utils import instantiate
    cfg = OmegaConf.load(config_path)
    cfg.model.net.in_s_channels = 1
    cfg.data.add_scalar_features = False
    if "include_global_token" not in cfg.data:
        if "num_global_tokens" in cfg.data:
            cfg.data.include_global_token = (cfg.data.num_global_tokens > 0)
        else:
            cfg.data.include_global_token = True  # LGATr always has a global token
    model = instantiate(cfg.model)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, cfg


def embed_lgatr_jet(p4_np, cfg, device):
    """Create the full embedding (particles + spurions + global token) for one jet."""
    n_phys = len(p4_np)
    x   = torch.tensor(p4_np, dtype=torch.float32, device=device)
    s   = torch.zeros(n_phys, 0, dtype=torch.float32, device=device)
    ptr = torch.tensor([0, n_phys], dtype=torch.long, device=device)
    return embed_tagging_data_into_ga(x, s, ptr, cfg.data)


def run_lgatr_emb(model, emb):
    """
    Run a pre-built embedding through full LGATr with hooks.
    Returns:
      hs_list  : list of NUM_LAYERS numpy arrays, each (n_tokens, s_ch)
      hmv_list : list of NUM_LAYERS numpy arrays, each (n_tokens, mv_ch, 16)
      logit    : float — scalar classification logit
    """
    hs_list  = [None] * NUM_LAYERS
    hmv_list = [None] * NUM_LAYERS

    def _make_hook(li):
        def hook(module, inp, out):
            h_mv, h_s = out
            hmv_list[li] = h_mv.squeeze(0).detach().cpu().numpy()
            hs_list[li]  = h_s.squeeze(0).detach().cpu().numpy()
        return hook

    hooks = []
    hooks.append(model.net.linear_in.register_forward_hook(_make_hook(0)))
    for li, blk in enumerate(model.net.blocks):
        hooks.append(blk.register_forward_hook(_make_hook(li + 1)))

    with torch.no_grad():
        out = model(emb)
    for h in hooks:
        h.remove()

    logit = float(out[0, 0].cpu())
    return hs_list, hmv_list, logit


def apply_rho_to_embedding(emb, rho_16_t, device):
    """
    Return a copy of emb with multivectors transformed by rho_16.
    emb["mv"] has shape (n_tokens, n_mv_ch, 16).
    rho_16_t: torch.float32 tensor (16, 16) on device.
    Applies ρ(Λ) to every multivector channel of every token:
      mv_new[..., j] = Σ_i mv[..., i] * rho_16[i, j]  →  mv @ rho_16
    Scalars are Lorentz-invariant and left unchanged.
    """
    emb_boost = {k: v for k, v in emb.items()}  # shallow copy
    emb_boost["mv"] = emb["mv"] @ rho_16_t       # (..., 16) @ (16,16)
    return emb_boost


def test_lgatr_equivariance(model, cfg, ds, n_jets, n_transforms, device, rng):
    """
    Equivariance test: embed once, then apply ρ(Λ) directly to the full
    embedding tensor (particles + spurions), so the beam reference is also
    transformed consistently.

    Returns:
      err_s_all     : (n_jets, n_transforms, NUM_LAYERS)  — h_s invariance errors
      err_mv_all    : (n_jets, n_transforms, NUM_LAYERS)  — h_mv equivariance errors (correct ρ)
      err_naive_all : (n_jets, n_transforms, NUM_LAYERS)  — h_mv errors if we predict no change
      logit_errs    : (n_jets, n_transforms)              — |logit(Λx)-logit(x)| / |logit(x)|
    """
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    err_s_all     = np.zeros((n_jets, n_transforms, NUM_LAYERS), dtype=np.float32)
    err_mv_all    = np.zeros((n_jets, n_transforms, NUM_LAYERS), dtype=np.float32)
    err_naive_all = np.zeros((n_jets, n_transforms, NUM_LAYERS), dtype=np.float32)
    logit_errs    = np.zeros((n_jets, n_transforms), dtype=np.float32)

    transforms = [sample_lorentz_transform(rng) for _ in range(n_transforms)]
    rho_tensors = [
        torch.tensor(rho_16, dtype=torch.float32, device=device)
        for (_, rho_16, _) in transforms
    ]

    jet_count = 0
    t0 = time.time()
    for batch in loader:
        if jet_count >= n_jets:
            break
        ptr = batch.ptr.cpu().numpy()
        p4  = batch.x[int(ptr[0]):int(ptr[1])].numpy()

        emb_orig = embed_lgatr_jet(p4, cfg, device)
        hs_orig, hmv_orig, logit_orig = run_lgatr_emb(model, emb_orig)

        for ti, ((u, rho_16, _), rho_t) in enumerate(zip(transforms, rho_tensors)):
            emb_boost = apply_rho_to_embedding(emb_orig, rho_t, device)
            hs_boost, hmv_boost, logit_boost = run_lgatr_emb(model, emb_boost)

            logit_errs[jet_count, ti] = abs(logit_boost - logit_orig) / (abs(logit_orig) + EPS)

            for li in range(NUM_LAYERS):
                err_s_all[jet_count, ti, li]     = rel_err(hs_boost[li], hs_orig[li])
                hmv_expected = apply_rho_to_hmv(hmv_orig[li], rho_16)
                err_mv_all[jet_count, ti, li]    = rel_err(hmv_boost[li], hmv_expected)
                err_naive_all[jet_count, ti, li] = rel_err(hmv_boost[li], hmv_orig[li])

        jet_count += 1
        if jet_count % 50 == 0:
            print(f"  LGATr: {jet_count}/{n_jets} jets  [{time.time()-t0:.0f}s]")

    return err_s_all, err_mv_all, err_naive_all, logit_errs


def test_boost_sweep(model, cfg, ds, gammas, n_jets, n_transforms, device, rng):
    """
    Sweep over pure boosts with specified Lorentz factors γ.
    Returns:
      sweep_mv    : (len(gammas), n_jets, n_transforms) — mean h_mv error across layers
      sweep_s     : (len(gammas), n_jets, n_transforms) — mean h_s error across layers
      sweep_logit : (len(gammas), n_jets, n_transforms) — logit invariance error
    """
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    n_g = len(gammas)
    sweep_mv    = np.zeros((n_g, n_jets, n_transforms), dtype=np.float32)
    sweep_s     = np.zeros((n_g, n_jets, n_transforms), dtype=np.float32)
    sweep_logit = np.zeros((n_g, n_jets, n_transforms), dtype=np.float32)

    jets = []
    for batch in loader:
        if len(jets) >= n_jets:
            break
        ptr = batch.ptr.cpu().numpy()
        jets.append(batch.x[int(ptr[0]):int(ptr[1])].numpy())

    t0 = time.time()
    for gi, gamma in enumerate(gammas):
        transforms = [sample_pure_boost(gamma, rng) for _ in range(n_transforms)]
        rho_tensors = [torch.tensor(r, dtype=torch.float32, device=device) for _, r, _ in transforms]

        for ji, p4 in enumerate(jets):
            emb_orig = embed_lgatr_jet(p4, cfg, device)
            hs_orig, hmv_orig, logit_orig = run_lgatr_emb(model, emb_orig)

            for ti, ((u, rho_16, _), rho_t) in enumerate(zip(transforms, rho_tensors)):
                emb_boost = apply_rho_to_embedding(emb_orig, rho_t, device)
                hs_boost, hmv_boost, logit_boost = run_lgatr_emb(model, emb_boost)

                sweep_logit[gi, ji, ti] = abs(logit_boost - logit_orig) / (abs(logit_orig) + EPS)
                layer_mv = [rel_err(hmv_boost[li], apply_rho_to_hmv(hmv_orig[li], rho_16))
                            for li in range(NUM_LAYERS)]
                layer_s  = [rel_err(hs_boost[li], hs_orig[li]) for li in range(NUM_LAYERS)]
                sweep_mv[gi, ji, ti] = np.mean(layer_mv)
                sweep_s[gi, ji, ti]  = np.mean(layer_s)

        print(f"  γ={gamma:5.1f}: h_mv={sweep_mv[gi].mean():.2e}  "
              f"h_s={sweep_s[gi].mean():.2e}  "
              f"logit={sweep_logit[gi].mean():.2e}  [{time.time()-t0:.0f}s]")

    return sweep_mv, sweep_s, sweep_logit


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_equivariance(err_s, err_mv_or_v, label_mv, title, save_path):
    """
    err_s, err_mv_or_v: (n_jets, n_transforms, NUM_LAYERS)
    Plots mean ± std across jets × transforms.
    Y-axis in log scale.
    """
    # Average over jets and transforms → mean per layer
    # Then compute std over jets×transforms pooled
    n_jets, n_tr, n_layers = err_s.shape
    flat_s  = err_s.reshape(-1, n_layers)    # (n_jets*n_tr, n_layers)
    flat_mv = err_mv_or_v.reshape(-1, n_layers)

    mean_s  = flat_s.mean(axis=0)
    std_s   = flat_s.std(axis=0)
    mean_mv = flat_mv.mean(axis=0)
    std_mv  = flat_mv.std(axis=0)

    layers = np.arange(n_layers)
    layer_labels = ["L0"] + [f"L{i}" for i in range(1, n_layers)]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.semilogy(layers, mean_s,  "b-o",  linewidth=2, markersize=5,
                label=f"h_s invariance")
    ax.fill_between(layers, np.maximum(mean_s - std_s, 1e-9), mean_s + std_s,
                    alpha=0.2, color="blue")
    ax.semilogy(layers, mean_mv, "r-s",  linewidth=2, markersize=5,
                label=label_mv)
    ax.fill_between(layers, np.maximum(mean_mv - std_mv, 1e-9), mean_mv + std_mv,
                    alpha=0.2, color="red")

    # Reference line for float32 machine epsilon
    ax.axhline(1.2e-7, color="gray", linestyle=":", linewidth=1.0,
               label="float32 machine ε (1.2e-7)")
    ax.axhline(1e-5,   color="silver", linestyle="--", linewidth=0.8,
               label="1e-5 threshold")

    ax.set_xticks(layers)
    ax.set_xticklabels(layer_labels, fontsize=9)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Relative equivariance error  (log scale)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")


def plot_boost_sweep(gammas, sweep_mv, sweep_s, sweep_logit, save_path):
    """Plot equivariance error vs Lorentz factor γ."""
    mean_mv    = sweep_mv.reshape(len(gammas), -1).mean(1)
    mean_s     = sweep_s.reshape(len(gammas), -1).mean(1)
    mean_logit = sweep_logit.reshape(len(gammas), -1).mean(1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(gammas, mean_mv,    "r-o", linewidth=2, markersize=7, label="h_mv equivariance")
    ax.loglog(gammas, mean_s,     "b-s", linewidth=2, markersize=7, label="h_s invariance")
    ax.loglog(gammas, mean_logit, "g-^", linewidth=2, markersize=7, label="logit invariance")
    ax.axhline(1.2e-7, color="gray",   linestyle=":", linewidth=1.0, label="float32 ε")
    ax.axhline(1e-3,   color="silver", linestyle="--", linewidth=0.8, label="1e-3")
    ax.set_xlabel("Lorentz boost factor γ", fontsize=12)
    ax.set_ylabel("Mean relative equivariance error", fontsize=12)
    ax.set_title("LGATr equivariance vs boost strength", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved boost sweep plot → {save_path}")


def write_summary(lgatr_s, lgatr_mv, lgatr_naive, logit_errs,
                  gammas, sweep_mv, sweep_s, sweep_logit,
                  n_jets, n_tr, save_path):
    layer_labels = ["L0(linear_in)"] + [f"L{i}" for i in range(1, NUM_LAYERS)]

    def _mean(arr):
        return arr.reshape(-1, NUM_LAYERS).mean(0)

    lines = [
        "Per-layer Lorentz Equivariance Test  —  Full LGATr",
        f"N_jets={n_jets}  N_transforms={n_tr}",
        "=" * 75,
        "",
        "Relative errors: ||h(Λx) - ρ(Λ)·h(x)|| / ||h(x)||",
        "  h_s correct : scalar invariance (should be ~0)",
        "  h_mv correct: multivector equivariance with correct ρ(Λ) (should be ~0)",
        "  h_mv naive  : if we predicted no change, i.e. h_mv(Λx)≈h_mv(x) (sanity baseline)",
        "",
        f"{'Layer':<18}  {'h_s correct':>13}  {'h_mv correct':>14}  {'h_mv naive':>12}",
    ]
    ms, mmv, mnaive = _mean(lgatr_s), _mean(lgatr_mv), _mean(lgatr_naive)
    for li in range(NUM_LAYERS):
        lines.append(
            f"  {layer_labels[li]:<16}  {ms[li]:>13.2e}  {mmv[li]:>14.2e}  {mnaive[li]:>12.2e}"
        )

    mean_logit = logit_errs.mean()
    lines += [
        "",
        f"Output-level logit invariance: |logit(Λx) - logit(x)| / |logit(x)|",
        f"  Mean over {n_jets} jets × {n_tr} transforms: {mean_logit:.2e}",
        "",
        "Boost sweep — pure boosts with increasing γ (mean error across all layers):",
        f"  {'γ':>6}  {'h_mv correct':>14}  {'h_s correct':>13}  {'logit':>8}",
    ]
    for gi, gamma in enumerate(gammas):
        mv_m = sweep_mv[gi].mean()
        s_m  = sweep_s[gi].mean()
        l_m  = sweep_logit[gi].mean()
        lines.append(f"  {gamma:>6.1f}  {mv_m:>14.2e}  {s_m:>13.2e}  {l_m:>8.2e}")

    text = "\n".join(lines)
    save_path.write_text(text)
    print(f"\nSaved summary → {save_path}")
    print(text)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lgatr-checkpoint",
                    required=True)
    ap.add_argument("--lgatr-config",
                    required=True)
    ap.add_argument("--data-path",    default="data/toptagging_full.npz")
    ap.add_argument("--output-dir",   default="results/equivariance")
    ap.add_argument("--n-jets",       type=int, default=200)
    ap.add_argument("--n-transforms", type=int, default=5)
    ap.add_argument("--n-jets-sweep", type=int, default=50,
                    help="Jets for boost sweep (faster than main test)")
    ap.add_argument("--gammas", type=float, nargs="+",
                    default=[1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0],
                    help="Lorentz boost factors for the pure-boost sweep")
    ap.add_argument("--seed",         type=int, default=42)
    args = ap.parse_args()

    GAMMAS = args.gammas

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    print(f"Device: {device}")
    print(f"n_jets={args.n_jets}  n_transforms={args.n_transforms}")

    ds = TopTaggingDataset(rescale_data=True)
    ds.load_data(args.data_path, mode="test")

    t_total = time.time()
    model_lgatr, cfg_lgatr = load_lgatr(
        args.lgatr_checkpoint, args.lgatr_config, device
    )

    # ── Main equivariance test (random transforms) ────────────────────────────
    print(f"\n{'='*60}\nPer-layer equivariance test ({args.n_jets} jets)...")
    lgatr_s, lgatr_mv, lgatr_naive, logit_errs = test_lgatr_equivariance(
        model_lgatr, cfg_lgatr, ds, args.n_jets, args.n_transforms, device, rng
    )
    print(f"  Mean logit invariance error: {logit_errs.mean():.2e}")

    # ── Boost sweep ────────────────────────────────────────────────
    print(f"\n{'='*60}\nBoost sweep γ={GAMMAS} ({args.n_jets_sweep} jets each)...")
    sweep_mv, sweep_s, sweep_logit = test_boost_sweep(
        model_lgatr, cfg_lgatr, ds, GAMMAS,
        args.n_jets_sweep, args.n_transforms, device, rng
    )

    del model_lgatr

    # ── Save ──────────────────────────────────────────────────────────────────
    np.savez(
        output_dir / "equivariance_errors.npz",
        lgatr_s=lgatr_s, lgatr_mv=lgatr_mv, lgatr_naive=lgatr_naive,
        logit_errs=logit_errs,
        gammas=np.array(GAMMAS),
        sweep_mv=sweep_mv, sweep_s=sweep_s, sweep_logit=sweep_logit,
    )

    plot_equivariance(
        lgatr_s, lgatr_mv,
        label_mv="h_mv equivariance (correct ρ(Λ))",
        title=f"Full LGATr — Per-layer Lorentz Equivariance  "
              f"({args.n_jets} jets × {args.n_transforms} transforms)",
        save_path=output_dir / "equivariance_lgatr.png",
    )
    plot_equivariance(
        lgatr_s, lgatr_naive,
        label_mv="h_mv naive baseline (no-transform prediction)",
        title=f"Full LGATr — Naive Baseline vs h_s Invariance  "
              f"({args.n_jets} jets × {args.n_transforms} transforms)",
        save_path=output_dir / "equivariance_naive.png",
    )
    plot_boost_sweep(
        GAMMAS, sweep_mv, sweep_s, sweep_logit,
        save_path=output_dir / "equivariance_boost_sweep.png",
    )
    write_summary(
        lgatr_s, lgatr_mv, lgatr_naive, logit_errs,
        GAMMAS, sweep_mv, sweep_s, sweep_logit,
        args.n_jets, args.n_transforms,
        output_dir / "equivariance_summary.txt",
    )

    print(f"\nTotal elapsed: {time.time()-t_total:.0f}s")
    print(f"All outputs in {output_dir}")


if __name__ == "__main__":
    main()
