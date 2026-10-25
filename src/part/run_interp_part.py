#!/usr/bin/env python3
"""
Combined attention map + linear probe extraction for ParT on TopTagging.

Runs a single forward pass per jet, capturing:
  1. Attention weights (8 main blocks × 8 heads × K × K)
  2. Hidden states (9 layers: embed + 8 blocks) for linear probes

Produces:
  - <outdir>/probe_data_part.npz   — same key structure as probe_data_slim.npz
  - <outdir>/avg_attn_all_Njets.png — averaged attention maps (2×4 grid per head)
  - <outdir>/ptrank_attn.npy/.png  — mean attention received per pT rank
  - <outdir>/attn_accum.npz        — raw attention sums for offline re-plotting
  - <outdir>/attn_density.png      — attention score density (off-diagonal)

Token layout: No special tokens. All P tokens are physical particles.
TOKEN_OFFSET = 0. N_LAYERS = 9 (L0 = after embed, L1..L8 = after each block).

Attention weights are per-head: (N_heads, P, P) per jet, averaged over jets.

Usage:
  cd path/to/particle_transformer
  CKPT=training/TopLandscape/ParT/20260418-194812_example_ParticleTransformer_ranger_lr0.001_batch512/net_best_epoch_state.pt
  CUDA_VISIBLE_DEVICES=1 nohup python3.10 run_interp_part.py \\
      --checkpoint $CKPT \\
      --data-path data/toptagging_full.npz \\
      --outdir results/part_interp/ \\
      --n-jets 10000 --k 50 \\
      > logs/part_interp.log 2>&1 &
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
from omegaconf import OmegaConf
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None

sys.path.insert(0, ".")
from networks.example_ParticleTransformer import ParticleTransformerWrapper

LLOCA_ROOT = Path(__file__).resolve().parent.parent / "lloca-experiments"
if LLOCA_ROOT.exists():
    sys.path.insert(0, str(LLOCA_ROOT))

from experiments.tagging.embedding import embed_tagging_data
from experiments.tagging.dataset import TopTaggingDataset

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

N_LAYERS   = 9    # L0 = after embed, L1..L8 = after each of 8 blocks
N_BLOCKS   = 8    # number of main (particle-to-particle) blocks
NUM_HEADS  = 8
EMBED_DIM  = 128

# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

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


def load_model(checkpoint, device, config_path=None):
    if config_path is not None:
        from hydra.utils import instantiate

        cfg = OmegaConf.load(config_path)
        model = instantiate(cfg.model)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        if any(k.startswith("net.module.") for k in state):
            state = {k.replace("net.module.", "net.", 1): v for k, v in state.items()}
        model.load_state_dict(state)
        model.to(device).eval()
        print(
            f"Loaded wrapper model ({sum(p.numel() for p in model.parameters()):,} params) "
            f"from {checkpoint}"
        )
        return model, cfg, True

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = _normalise_checkpoint_state(state)
    num_classes = int(state["mod.fc.0.weight"].shape[0])
    cfg = dict(
        input_dim=7,       # log_pt, log_e, log_pt_rel, log_e_rel, dR, deta, dphi
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
    print(
        f"Loaded model ({sum(p.numel() for p in model.parameters()):,} params, "
        f"num_classes={num_classes})"
    )
    return model, None, False

# ──────────────────────────────────────────────────────────────────────────────
# Patch MHA to return per-head weights
# ──────────────────────────────────────────────────────────────────────────────

def _patch_attn_perhead(mha_module):
    """Force nn.MultiheadAttention to return (N, num_heads, T, S) attention weights."""
    orig = mha_module.forward
    def patched(query, key, value, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False
        return orig(query, key, value, **kwargs)
    mha_module.forward = patched

# ──────────────────────────────────────────────────────────────────────────────
# Data preprocessing from NPZ
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_jet(E, px, py, pz, k_max):
    """
    Constructs ParT inputs from raw 4-momenta.
    Features standardized per data/TopLandscape/top_kin.yaml.

    Returns:
        features: (1, 7, P) tensor
        vectors:  (1, 4, P) tensor  [px, py, pz, energy]
        mask:     (1, 1, P) bool tensor
        p4:       (P, 4) numpy array  [E, px, py, pz]
    """
    pt  = np.sqrt(px**2 + py**2)
    eta = np.arcsinh(pz / (pt + 1e-10))
    phi = np.arctan2(py, px)

    jet_px = px.sum(); jet_py = py.sum()
    jet_pt  = np.sqrt(jet_px**2 + jet_py**2)
    jet_eta = np.arcsinh(pz.sum() / (jet_pt + 1e-10))
    jet_phi = np.arctan2(jet_py, jet_px)
    jet_E   = E.sum()

    deta = eta - jet_eta
    dphi = ((phi - jet_phi) + np.pi) % (2 * np.pi) - np.pi
    dR   = np.sqrt(deta**2 + dphi**2)

    # standardization from top_kin.yaml: (x - subtract_by) * multiply_by
    log_pt     = (np.log(pt + 1e-10) - 1.7) * 0.7
    log_e      = (np.log(E  + 1e-10) - 2.0) * 0.7
    log_pt_rel = (np.log(pt / (jet_pt + 1e-10) + 1e-10) - (-4.7)) * 0.7
    log_e_rel  = (np.log(E  / (jet_E  + 1e-10) + 1e-10) - (-4.7)) * 0.7
    dR_feat    = (dR - 0.2) * 4.0

    feats = np.stack([log_pt, log_e, log_pt_rel, log_e_rel, dR_feat, deta, dphi])  # (7, P)
    feats = np.clip(feats, -5, 5)

    points = np.stack([deta, dphi])  # (2, P) — pf_points

    vecs = np.stack([px, py, pz, E])  # (4, P)  — [px, py, pz, energy] per ParT convention

    features_t = torch.tensor(feats[np.newaxis],   dtype=torch.float32)   # (1, 7, P)
    points_t   = torch.tensor(points[np.newaxis],  dtype=torch.float32)   # (1, 2, P)
    vectors_t  = torch.tensor(vecs[np.newaxis],    dtype=torch.float32)   # (1, 4, P)
    mask_t     = torch.ones(1, 1, len(E), dtype=torch.bool)

    p4 = np.column_stack([E, px, py, pz]).astype(np.float32)  # (P, 4)
    return features_t, points_t, vectors_t, mask_t, p4

# ──────────────────────────────────────────────────────────────────────────────
# Physics label helpers (same as run_probes_slim.py)
# ──────────────────────────────────────────────────────────────────────────────

def _pt(p4):   return np.sqrt(p4[:, 1]**2 + p4[:, 2]**2)
def _eta(p4):  return np.arcsinh(p4[:, 3] / (_pt(p4) + 1e-10))
def _phi(p4):  return np.arctan2(p4[:, 2], p4[:, 1])

def jet_mass(p4):
    j = p4.sum(0)
    return float(np.sqrt(max(j[0]**2 - j[1]**2 - j[2]**2 - j[3]**2, 0.)))

def dR_to_jet_axis(p4):
    j = p4.sum(0); pt_j = np.sqrt(j[1]**2 + j[2]**2)
    ej = float(np.arcsinh(j[3] / (pt_j + 1e-10)))
    pj = float(np.arctan2(j[2], j[1]))
    de = _eta(p4) - ej
    dp = (_phi(p4) - pj + np.pi) % (2*np.pi) - np.pi
    return np.sqrt(de**2 + dp**2).astype(np.float32)

def pt_rank_quartile(p4):
    n = len(p4); pt = _pt(p4)
    rnk = np.argsort(np.argsort(-pt))
    return (rnk * 4 // n).clip(0, 3).astype(np.int32)

def _kt_merge(cp, ce, cp2, target):
    while len(cp) > target:
        n = len(cp)
        pi = cp[:, None]; pj = cp[None, :]
        de = ce[:, None] - ce[None, :]
        dp = cp2[:, None] - cp2[None, :]
        dp = (dp + np.pi) % (2*np.pi) - np.pi
        d = np.minimum(pi**2, pj**2) * (de**2 + dp**2)
        np.fill_diagonal(d, np.inf)
        flat = np.argmin(d); i, j = flat // n, flat % n
        if i > j: i, j = j, i
        pt_new = cp[i] + cp[j]; wj = cp[j] / (pt_new + 1e-10)
        eta_new = ce[i] + wj * (ce[j] - ce[i])
        dpij = (cp2[j] - cp2[i] + np.pi) % (2*np.pi) - np.pi
        phi_new = cp2[i] + wj * dpij
        m = np.ones(n, bool); m[i] = m[j] = False
        cp = np.append(cp[m], pt_new); ce = np.append(ce[m], eta_new)
        cp2 = np.append(cp2[m], phi_new)
    return cp, ce, cp2

def _tau_axes(pt, eta, phi, axes, R=0.8):
    d0 = pt.sum() * R + 1e-10
    ae = np.array([a[0] for a in axes]); ap = np.array([a[1] for a in axes])
    tau = sum(pt[i] * np.sqrt((eta[i]-ae)**2 + ((phi[i]-ap+np.pi)%(2*np.pi)-np.pi)**2).min()
              for i in range(len(pt)))
    return tau / d0

def nsubjettiness(p4, R=0.8):
    pt = _pt(p4).astype(float); eta = _eta(p4).astype(float); phi = _phi(p4).astype(float)
    axes_at = {}; cp, ce, cp2 = pt.copy(), eta.copy(), phi.copy()
    while len(cp) > 1:
        if len(cp) <= 3:
            axes_at[len(cp)] = list(zip(ce.copy(), cp2.copy()))
        cp, ce, cp2 = _kt_merge(cp, ce, cp2, len(cp) - 1)
    axes_at[1] = [(ce[0], cp2[0])]
    for n in (2, 3):
        if n not in axes_at: axes_at[n] = axes_at[n-1]
    t1 = _tau_axes(pt, eta, phi, axes_at[1], R)
    t2 = _tau_axes(pt, eta, phi, axes_at[2], R)
    t3 = _tau_axes(pt, eta, phi, axes_at[3], R)
    return float(t1), float(t2), float(t3)

# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_avg_attn(avg_attn, n_jets, k, outdir):
    """Plot 8-head averaged attention maps in a 2×4 grid."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f"ParT  TopTagging  Averaged attention  (n={n_jets}, K={k})", fontsize=14)
    for h in range(NUM_HEADS):
        ax = axes[h // 4, h % 4]
        if sns is not None:
            sns.heatmap(avg_attn[h], ax=ax, cmap="viridis", cbar=True,
                        square=True, xticklabels=False, yticklabels=False)
        else:
            im = ax.imshow(avg_attn[h], cmap="viridis", aspect="equal", origin="lower")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_title(f"Head {h}", fontsize=11)
    plt.tight_layout()
    path = outdir / f"avg_attn_all_{n_jets}jets.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_ptrank_attn(ptrank_mean, k, outdir):
    """Mean attention received vs pT rank (rank 0 = highest pT)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(k), ptrank_mean, "o-", color="#d62728", linewidth=2, markersize=4)
    ax.set_xlabel("pT rank (0 = highest pT)", fontsize=12)
    ax.set_ylabel("Mean attention received", fontsize=12)
    ax.set_title("ParT  TopTagging  Attention by pT rank  (head+query averaged)", fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = outdir / "ptrank_attn.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_attn_density(off_diag_scores, k, outdir):
    bins = np.linspace(0, 1, 51)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(off_diag_scores, bins=bins, density=True, color="steelblue", edgecolor="black")
    ax.set_yscale("log")
    ax.set_xlabel("Attention score", fontsize=13)
    ax.set_ylabel("Density (log scale)", fontsize=13)
    ax.set_title(f"ParT  TopTagging  Off-diagonal attention scores  K={k}", fontsize=12)
    plt.tight_layout()
    path = outdir / "attn_density.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

# ──────────────────────────────────────────────────────────────────────────────
# Main extraction loop
# ──────────────────────────────────────────────────────────────────────────────

def run(model, data, n_jets, k, outdir, device):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    kinematics = data["kinematics_test"]  # (N, 200, 4): E, px, py, pz
    labels_raw = data["labels_test"]      # (N,)
    N_total    = min(n_jets, len(labels_raw))

    # ── Patch MHA for per-head weights ────────────────────────────────────────
    for blk in model.mod.blocks:
        _patch_attn_perhead(blk.attn)

    # ── Hook storage ──────────────────────────────────────────────────────────
    hs_captured  = [None] * N_LAYERS   # L0..L8 hidden states
    attn_captured = [None] * N_BLOCKS  # per-block attention (1, H, P, P)

    def _make_hs_hook(li):
        def hook(module, inp, out):
            # out: (P, 1, embed_dim) for embed and blocks
            hs_captured[li] = out.detach().cpu()
        return hook

    def _make_attn_hook(bi):
        def hook(module, inp, out):
            # out: (attn_output, attn_weights)  attn_weights: (1, H, P, P)
            if isinstance(out, tuple) and len(out) == 2 and out[1] is not None:
                attn_captured[bi] = out[1].detach().cpu()
        return hook

    hooks = []
    hooks.append(model.mod.embed.register_forward_hook(_make_hs_hook(0)))
    for bi, blk in enumerate(model.mod.blocks):
        hooks.append(blk.register_forward_hook(_make_hs_hook(bi + 1)))
        hooks.append(blk.attn.register_forward_hook(_make_attn_hook(bi)))

    # ── Accumulators ──────────────────────────────────────────────────────────
    attn_sum   = np.zeros((N_BLOCKS, NUM_HEADS, k, k))  # summed over jets
    attn_count = 0
    ptrank_sum = np.zeros(k)   # attention received per pT rank (sum)
    off_diag_all = []

    # Probe buffers
    jet_reps_buf  = [[] for _ in range(N_LAYERS)]
    part_reps_buf = [[] for _ in range(N_LAYERS)]

    jet_labels_l    = []
    jet_mass_l      = []
    jet_mult_l      = []
    jet_tau1_l      = []
    jet_tau2_l      = []
    jet_tau3_l      = []
    part_jet_idx_l  = []
    part_pt_l       = []
    part_eta_l      = []
    part_phi_l      = []
    part_E_l        = []
    part_quartile_l = []
    part_dR_l       = []

    t0 = time.time()

    try:
        for ji in range(N_total):
            raw = kinematics[ji]                    # (200, 4): E, px, py, pz
            E_all, px_all, py_all, pz_all = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]

            is_real = (E_all != 0) | (px_all != 0) | (py_all != 0) | (pz_all != 0)
            n_phys  = int(is_real.sum())
            if n_phys < 3:
                continue

            idx  = np.where(is_real)[0][:k]         # first k real particles
            n_use = len(idx)
            E_j, px_j, py_j, pz_j = E_all[idx], px_all[idx], py_all[idx], pz_all[idx]

            # Forward pass
            feat_t, pts_t, vec_t, mask_t, p4 = preprocess_jet(E_j, px_j, py_j, pz_j, k)
            feat_t  = feat_t.to(device)
            pts_t   = pts_t.to(device)
            vec_t   = vec_t.to(device)
            mask_t  = mask_t.to(device)

            with torch.no_grad():
                _ = model(pts_t, feat_t, lorentz_vectors=vec_t, mask=mask_t)

            # ── Physics labels ─────────────────────────────────────────────
            label = int(labels_raw[ji])
            mass  = jet_mass(p4)
            tau1, tau2, tau3 = nsubjettiness(p4)

            pt_p      = _pt(p4).astype(np.float32)
            eta_p     = _eta(p4).astype(np.float32)
            phi_p     = _phi(p4).astype(np.float32)
            E_p       = p4[:, 0].astype(np.float32)
            quartiles = pt_rank_quartile(p4)
            dR_p      = dR_to_jet_axis(p4)

            jet_labels_l.append(label)
            jet_mass_l.append(mass)
            jet_mult_l.append(n_phys)
            jet_tau1_l.append(tau1)
            jet_tau2_l.append(tau2)
            jet_tau3_l.append(tau3)
            part_jet_idx_l.extend([attn_count] * n_use)
            part_pt_l.extend(pt_p.tolist())
            part_eta_l.extend(eta_p.tolist())
            part_phi_l.extend(phi_p.tolist())
            part_E_l.extend(E_p.tolist())
            part_quartile_l.extend(quartiles.tolist())
            part_dR_l.extend(dR_p.tolist())

            # ── Hidden state representations (probes) ─────────────────────
            for li in range(N_LAYERS):
                h = hs_captured[li]  # (P, 1, 128)
                if h is None:
                    continue
                h_np = h.squeeze(1).numpy().astype(np.float32)  # (P, 128)

                # jet-level: mean pool over real particles
                jet_rep  = h_np.mean(axis=0)
                part_rep = h_np[:n_use]

                jet_reps_buf[li].append(jet_rep)
                part_reps_buf[li].append(part_rep)

            # ── Attention maps (only for jets with exactly k particles) ────
            pt_rank   = np.argsort(np.argsort(-pt_p))  # rank[i] = pT rank of particle i
            do_attn   = (n_use == k)   # attention grid is (k,k) — skip smaller jets

            if do_attn:
                for bi in range(N_BLOCKS):
                    a = attn_captured[bi]   # (1, H, k, k)
                    if a is None:
                        continue
                    a_np = a[0].numpy()     # (H, k, k)

                    attn_sum[bi] += a_np
                    if bi == N_BLOCKS - 1:
                        mask = ~np.eye(k, dtype=bool)
                        off_diag_all.append(a_np[:, mask].flatten())

                    # pT-rank: final block only
                    if bi == N_BLOCKS - 1:
                        head_avg     = a_np.mean(0)         # (k, k)
                        recv_by_pos  = head_avg.mean(0)     # mean attn received per position
                        for pos_i in range(k):
                            ptrank_sum[pt_rank[pos_i]] += recv_by_pos[pos_i]

                attn_count += 1

            if attn_count % 500 == 0:
                elapsed = time.time() - t0
                print(f"  [{elapsed:5.0f}s]  {attn_count}/{N_total} jets")

    finally:
        for h in hooks:
            h.remove()

    print(f"\nDone: {attn_count} jets in {time.time()-t0:.0f}s")

    # ── Assemble probe data ───────────────────────────────────────────────────
    tau1 = np.array(jet_tau1_l, dtype=np.float32)
    tau2 = np.array(jet_tau2_l, dtype=np.float32)
    tau3 = np.array(jet_tau3_l, dtype=np.float32)

    probe_data = {
        "jet_reps"                 : np.stack([np.stack(jet_reps_buf[li])  for li in range(N_LAYERS)]),
        "particle_reps"            : np.stack([np.vstack(part_reps_buf[li]) for li in range(N_LAYERS)]),
        "particle_jet_idx"         : np.array(part_jet_idx_l,  dtype=np.int32),
        "jet_labels"               : np.array(jet_labels_l,    dtype=np.int32),
        "jet_mass"                 : np.array(jet_mass_l,       dtype=np.float32),
        "jet_multiplicity"         : np.array(jet_mult_l,       dtype=np.int32),
        "jet_tau1"                 : tau1,
        "jet_tau2"                 : tau2,
        "jet_tau3"                 : tau3,
        "jet_tau21"                : tau2 / (tau1 + 1e-10),
        "particle_pt"              : np.array(part_pt_l,        dtype=np.float32),
        "particle_eta"             : np.array(part_eta_l,       dtype=np.float32),
        "particle_phi"             : np.array(part_phi_l,       dtype=np.float32),
        "particle_E"               : np.array(part_E_l,         dtype=np.float32),
        "particle_pt_rank_quartile": np.array(part_quartile_l,  dtype=np.int32),
        "particle_dR_to_axis"      : np.array(part_dR_l,        dtype=np.float32),
    }

    np.savez_compressed(outdir / "probe_data_part.npz", **probe_data)
    print(f"Saved probe data → {outdir}/probe_data_part.npz")
    for key, val in probe_data.items():
        print(f"  {key:35s}: {str(val.shape):25s}  {val.dtype}")

    # ── Attention plots ───────────────────────────────────────────────────────
    if attn_count > 0:
        avg_attn_final = attn_sum[-1] / attn_count  # last block, (H, k, k)
        plot_avg_attn(avg_attn_final, attn_count, k, outdir)

        # save per-block averages
        avg_attn_all = attn_sum / attn_count  # (N_BLOCKS, H, k, k)
        np.savez(outdir / "attn_accum.npz",
                 attn_sum=attn_sum,
                 attn_count=np.array(attn_count),
                 avg_attn_all=avg_attn_all)
        print(f"Saved attn_accum → {outdir}/attn_accum.npz")

        # pT-rank attention
        ptrank_mean = ptrank_sum / attn_count
        np.save(outdir / "ptrank_attn.npy", ptrank_mean)
        plot_ptrank_attn(ptrank_mean, k, outdir)

        # density plot
        if off_diag_all:
            flat = np.concatenate(off_diag_all)
            plot_attn_density(flat, k, outdir)


def run_wrapper(model, cfg, data_path, n_jets, k, outdir, device):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    momentum_float64 = bool(cfg.data.get("momentum_float64", True))
    ds = TopTaggingDataset()
    ds.load_data(data_path, mode="test", momentum_float64=momentum_float64)
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    n_target = min(n_jets, len(ds))

    core = getattr(model, "net", model)
    core = getattr(core, "mod", core)

    for blk in core.blocks:
        _patch_attn_perhead(blk.attn)

    hs_captured = [None] * N_LAYERS
    attn_captured = [None] * N_BLOCKS

    def _make_hs_hook(li):
        def hook(module, inp, out):
            hs_captured[li] = out.detach().cpu()
        return hook

    def _make_attn_hook(bi):
        def hook(module, inp, out):
            if isinstance(out, tuple) and len(out) == 2 and out[1] is not None:
                attn_captured[bi] = out[1].detach().cpu()
        return hook

    hooks = []
    hooks.append(core.embed.register_forward_hook(_make_hs_hook(0)))
    for bi, blk in enumerate(core.blocks):
        hooks.append(blk.register_forward_hook(_make_hs_hook(bi + 1)))
        hooks.append(blk.attn.register_forward_hook(_make_attn_hook(bi)))

    attn_sum = np.zeros((N_BLOCKS, NUM_HEADS, k, k))
    attn_count = 0
    ptrank_sum = np.zeros(k)
    off_diag_all = []

    jet_reps_buf = [[] for _ in range(N_LAYERS)]
    part_reps_buf = [[] for _ in range(N_LAYERS)]

    jet_labels_l = []
    jet_mass_l = []
    jet_mult_l = []
    jet_tau1_l = []
    jet_tau2_l = []
    jet_tau3_l = []
    part_jet_idx_l = []
    part_pt_l = []
    part_eta_l = []
    part_phi_l = []
    part_E_l = []
    part_quartile_l = []
    part_dR_l = []

    jet_count = 0
    t0 = time.time()

    try:
        for batch in loader:
            if jet_count >= n_target:
                break

            ptr = batch.ptr.cpu().numpy()
            n_this = min(len(ptr) - 1, n_target - jet_count)

            for bi in range(n_this):
                start, end = int(ptr[bi]), int(ptr[bi + 1])
                n_phys_raw = end - start
                if n_phys_raw < 3:
                    continue

                n_use = min(n_phys_raw, k)
                p4_full = batch.x[start:end].numpy()
                p4 = p4_full[:n_use]

                scalars_in = (
                    batch.scalars[start:end].clone()
                    if hasattr(batch, "scalars") and batch.scalars is not None
                    else torch.zeros(n_phys_raw, 0, dtype=torch.float32)
                )
                sj = Data(
                    x=batch.x[start:end].clone(),
                    scalars=scalars_in,
                    ptr=torch.tensor([0, n_phys_raw], dtype=torch.long),
                    label=batch.label[bi : bi + 1].clone(),
                )

                for li in range(N_LAYERS):
                    hs_captured[li] = None
                for li in range(N_BLOCKS):
                    attn_captured[li] = None

                with torch.no_grad():
                    sj_dev = sj.to(device)
                    scalars_dev = (
                        sj_dev.scalars
                        if hasattr(sj_dev, "scalars") and sj_dev.scalars is not None
                        else torch.zeros(sj_dev.x.shape[0], 0, device=device)
                    )
                    embedding = embed_tagging_data(sj_dev.x, scalars_dev, sj_dev.ptr, cfg.data)
                    embedding["num_graphs"] = 1
                    _ = model(embedding)

                jet_idx = jet_count

                label = int(batch.label[bi].item())
                mass = jet_mass(p4)
                tau1, tau2, tau3 = nsubjettiness(p4)

                pt_p = _pt(p4).astype(np.float32)
                eta_p = _eta(p4).astype(np.float32)
                phi_p = _phi(p4).astype(np.float32)
                E_p = p4[:, 0].astype(np.float32)
                quartiles = pt_rank_quartile(p4)
                dR_p = dR_to_jet_axis(p4)

                jet_labels_l.append(label)
                jet_mass_l.append(mass)
                jet_mult_l.append(n_phys_raw)
                jet_tau1_l.append(tau1)
                jet_tau2_l.append(tau2)
                jet_tau3_l.append(tau3)
                part_jet_idx_l.extend([jet_idx] * n_use)
                part_pt_l.extend(pt_p.tolist())
                part_eta_l.extend(eta_p.tolist())
                part_phi_l.extend(phi_p.tolist())
                part_E_l.extend(E_p.tolist())
                part_quartile_l.extend(quartiles.tolist())
                part_dR_l.extend(dR_p.tolist())

                for li in range(N_LAYERS):
                    h = hs_captured[li]
                    if h is None:
                        continue
                    h_np = h.squeeze(1).numpy().astype(np.float32)
                    jet_reps_buf[li].append(h_np.mean(axis=0))
                    part_reps_buf[li].append(h_np[:n_use])

                pt_rank = np.argsort(np.argsort(-pt_p))
                do_attn = n_use == k
                if do_attn:
                    for blk_idx in range(N_BLOCKS):
                        a = attn_captured[blk_idx]
                        if a is None:
                            continue
                        a_np = a[0].numpy()
                        attn_sum[blk_idx] += a_np
                        if blk_idx == N_BLOCKS - 1:
                            mask = ~np.eye(k, dtype=bool)
                            off_diag_all.append(a_np[:, mask].flatten())
                            head_avg = a_np.mean(0)
                            recv_by_pos = head_avg.mean(0)
                            for pos_i in range(k):
                                ptrank_sum[pt_rank[pos_i]] += recv_by_pos[pos_i]
                    attn_count += 1

                jet_count += 1
                if jet_count % 500 == 0:
                    elapsed = time.time() - t0
                    print(f"  [{elapsed:5.0f}s]  {jet_count}/{n_target} jets")

    finally:
        for h in hooks:
            h.remove()

    print(f"\nDone: {jet_count} jets in {time.time()-t0:.0f}s")

    tau1 = np.array(jet_tau1_l, dtype=np.float32)
    tau2 = np.array(jet_tau2_l, dtype=np.float32)
    tau3 = np.array(jet_tau3_l, dtype=np.float32)

    probe_data = {
        "jet_reps": np.stack([np.stack(jet_reps_buf[li]) for li in range(N_LAYERS)]),
        "particle_reps": np.stack([np.vstack(part_reps_buf[li]) for li in range(N_LAYERS)]),
        "particle_jet_idx": np.array(part_jet_idx_l, dtype=np.int32),
        "jet_labels": np.array(jet_labels_l, dtype=np.int32),
        "jet_mass": np.array(jet_mass_l, dtype=np.float32),
        "jet_multiplicity": np.array(jet_mult_l, dtype=np.int32),
        "jet_tau1": tau1,
        "jet_tau2": tau2,
        "jet_tau3": tau3,
        "jet_tau21": tau2 / (tau1 + 1e-10),
        "particle_pt": np.array(part_pt_l, dtype=np.float32),
        "particle_eta": np.array(part_eta_l, dtype=np.float32),
        "particle_phi": np.array(part_phi_l, dtype=np.float32),
        "particle_E": np.array(part_E_l, dtype=np.float32),
        "particle_pt_rank_quartile": np.array(part_quartile_l, dtype=np.int32),
        "particle_dR_to_axis": np.array(part_dR_l, dtype=np.float32),
    }

    np.savez_compressed(outdir / "probe_data_part.npz", **probe_data)
    print(f"Saved probe data → {outdir}/probe_data_part.npz")
    for key, val in probe_data.items():
        print(f"  {key:35s}: {str(val.shape):25s}  {val.dtype}")

    if attn_count > 0:
        avg_attn_final = attn_sum[-1] / attn_count
        plot_avg_attn(avg_attn_final, attn_count, k, outdir)

        avg_attn_all = attn_sum / attn_count
        np.savez(
            outdir / "attn_accum.npz",
            attn_sum=attn_sum,
            attn_count=np.array(attn_count),
            avg_attn_all=avg_attn_all,
        )
        print(f"Saved attn_accum → {outdir}/attn_accum.npz")

        ptrank_mean = ptrank_sum / attn_count
        np.save(outdir / "ptrank_attn.npy", ptrank_mean)
        plot_ptrank_attn(ptrank_mean, k, outdir)

        if off_diag_all:
            flat = np.concatenate(off_diag_all)
            plot_attn_density(flat, k, outdir)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ckpt_default = (
        "training/TopLandscape/ParT/20260418-194812_example_ParticleTransformer_ranger_lr0.001_batch512"
        "/net_best_epoch_state.pt"
    )
    ap.add_argument("--checkpoint", default=ckpt_default)
    ap.add_argument("--config", default=None,
                    help="lloca-experiments config.yaml for wrapper-era ParT checkpoints")
    ap.add_argument("--data-path",  default="data/toptagging_full.npz")
    ap.add_argument("--outdir",     default="results/part_interp/")
    ap.add_argument("--n-jets",     type=int, default=10000)
    ap.add_argument("--k",          type=int, default=50)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, cfg, use_wrapper = load_model(args.checkpoint, device, config_path=args.config)
    if use_wrapper:
        run_wrapper(
            model,
            cfg,
            data_path=args.data_path,
            n_jets=args.n_jets,
            k=args.k,
            outdir=args.outdir,
            device=device,
        )
    else:
        print(f"Loading {args.data_path} ...")
        _npz = np.load(args.data_path, mmap_mode="r")
        data = {"kinematics_test": _npz["kinematics_test"],
                "labels_test":     _npz["labels_test"]}
        run(model, data, n_jets=args.n_jets, k=args.k, outdir=args.outdir, device=device)


if __name__ == "__main__":
    main()
