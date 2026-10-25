#!/usr/bin/env python
"""
Extract per-layer scalar representations from the full LGATr model for linear probing.

Registers forward hooks on model.net.linear_in (layer 0) and each of the 12
GATrBlocks (layers 1-12), capturing the scalar output h_s after each layer.

Token layout (full LGATr, token_offset=3):
  index 0          : global token          → jet-level representation
  index 1          : xy-plane spurion
  index 2          : time spurion
  index 3..3+n-1   : physical particles    → particle-level representations

Saves results/probe_data_lgatr.npz with:
  jet_reps                (n_layers, n_jets, 32)          float32
  particle_reps           (n_layers, total_particles, 32) float32
  particle_jet_idx        (total_particles,)               int32
  jet_labels              (n_jets,)                        int32
  jet_mass                (n_jets,)                        float32
  jet_multiplicity        (n_jets,)                        int32
  jet_tau1/2/3            (n_jets,)                        float32
  jet_tau21               (n_jets,)                        float32
  particle_pt/eta/phi/E   (total_particles,)               float32
  particle_pt_rank_quartile (total_particles,)             int32
  particle_dR_to_axis     (total_particles,)               float32

Usage:
  cd path/to/lorentz-gatr
  CUDA_VISIBLE_DEVICES=0 nohup venv/bin/python run_probes_lgatr.py \\
      --checkpoint results/models/topt_GATr_7327_run0_it200000.pt \\
      --config     runs/topt/GATr_7327/config.yaml \\
      --data-path  data/toptagging_full.npz \\
      --output     results/probe_data_lgatr.npz \\
      --n-jets     10000 \\
      > results/probe_lgatr.log 2>&1 &
"""

import argparse
import sys
import time
import numpy as np
import torch
from pathlib import Path
from omegaconf import OmegaConf
from torch_geometric.loader import DataLoader

sys.path.insert(0, ".")

from experiments.tagging.embedding import embed_tagging_data
from experiments.tagging.dataset import TopTaggingDataset

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

N_LAYERS     = 13  # 0 = after linear_in,  1..12 = after each GATrBlock
K_MAX        = 50  # max particles per jet to store

# ──────────────────────────────────────────────────────────────────────────────
# Physics label helpers  (all operate on raw un-rescaled 4-momenta)
# ──────────────────────────────────────────────────────────────────────────────

def _pt(p4):
    return np.sqrt(p4[:, 1]**2 + p4[:, 2]**2)

def _eta(p4):
    pt = _pt(p4)
    return np.arcsinh(p4[:, 3] / (pt + 1e-10))

def _phi(p4):
    return np.arctan2(p4[:, 2], p4[:, 1])

def jet_mass(p4):
    """Invariant mass of the jet from summed 4-momenta."""
    j = p4.sum(axis=0)
    m2 = j[0]**2 - j[1]**2 - j[2]**2 - j[3]**2
    return float(np.sqrt(max(m2, 0.0)))

def dR_to_jet_axis(p4):
    """ΔR of each particle to the jet axis (defined as sum of all 4-momenta)."""
    j    = p4.sum(axis=0)
    pt_j = np.sqrt(j[1]**2 + j[2]**2)
    eta_j = float(np.arcsinh(j[3] / (pt_j + 1e-10)))
    phi_j = float(np.arctan2(j[2], j[1]))
    deta = _eta(p4) - eta_j
    dphi = (_phi(p4) - phi_j + np.pi) % (2 * np.pi) - np.pi
    return np.sqrt(deta**2 + dphi**2).astype(np.float32)

def pt_rank_quartile(p4):
    """0 = hardest quartile, 3 = softest (within jet, 0-indexed)."""
    n   = len(p4)
    pt  = _pt(p4)
    rnk = np.argsort(np.argsort(-pt))   # 0 = hardest
    return (rnk * 4 // n).clip(0, 3).astype(np.int32)

def _exclusive_kt_merge(curr_pt, curr_eta, curr_phi, target_n):
    """
    Run exclusive kT clustering until `target_n` pseudojets remain.
    Returns (pt, eta, phi) arrays of the remaining pseudojets.
    Uses vectorised numpy; O(n²) per step.
    """
    while len(curr_pt) > target_n:
        n    = len(curr_pt)
        pt_i = curr_pt[:, None];  pt_j = curr_pt[None, :]
        de   = curr_eta[:, None] - curr_eta[None, :]
        dp   = curr_phi[:, None] - curr_phi[None, :]
        dp   = (dp + np.pi) % (2 * np.pi) - np.pi
        d    = np.minimum(pt_i**2, pt_j**2) * (de**2 + dp**2)
        np.fill_diagonal(d, np.inf)

        flat = np.argmin(d);  i, j = flat // n, flat % n
        if i > j: i, j = j, i

        pt_new  = curr_pt[i] + curr_pt[j]
        wj      = curr_pt[j] / (pt_new + 1e-10)
        eta_new = curr_eta[i] + wj * (curr_eta[j] - curr_eta[i])
        dpij    = (curr_phi[j] - curr_phi[i] + np.pi) % (2 * np.pi) - np.pi
        phi_new = curr_phi[i] + wj * dpij

        mask         = np.ones(n, dtype=bool)
        mask[i]      = False;  mask[j] = False
        curr_pt      = np.append(curr_pt[mask],  pt_new)
        curr_eta     = np.append(curr_eta[mask], eta_new)
        curr_phi     = np.append(curr_phi[mask], phi_new)

    return curr_pt, curr_eta, curr_phi

def _tau_from_axes(pt, eta, phi, axes, R=0.8):
    """τ_N = Σ pT_i min_k(ΔR_{i,k})  /  (Σ pT_i · R)"""
    d0  = pt.sum() * R + 1e-10
    tau = 0.0
    ax_eta = np.array([a[0] for a in axes])
    ax_phi = np.array([a[1] for a in axes])
    for i in range(len(pt)):
        de  = eta[i] - ax_eta
        dp  = (phi[i] - ax_phi + np.pi) % (2 * np.pi) - np.pi
        tau += pt[i] * np.sqrt(de**2 + dp**2).min()
    return tau / d0

def nsubjettiness(p4, R=0.8):
    """
    Compute τ₁, τ₂, τ₃ using exclusive kT axes.
    Runs the algorithm once, saving axis states at N=3, 2, 1.
    Returns (tau1, tau2, tau3) as Python floats.
    """
    pt  = _pt(p4).astype(float)
    eta = _eta(p4).astype(float)
    phi = _phi(p4).astype(float)

    axes_at = {}
    cp, ce, cphi = pt.copy(), eta.copy(), phi.copy()

    while len(cp) > 1:
        if len(cp) <= 3:
            axes_at[len(cp)] = list(zip(ce.copy(), cphi.copy()))
        cp, ce, cphi = _exclusive_kt_merge(cp, ce, cphi, len(cp) - 1)

    axes_at[1] = [(ce[0], cphi[0])]

    # fill gaps if jet had < 3 particles
    for n in (2, 3):
        if n not in axes_at:
            axes_at[n] = axes_at[n - 1]

    tau1 = _tau_from_axes(pt, eta, phi, axes_at[1], R)
    tau2 = _tau_from_axes(pt, eta, phi, axes_at[2], R)
    tau3 = _tau_from_axes(pt, eta, phi, axes_at[3], R)
    return float(tau1), float(tau2), float(tau3)

# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path, config_path, device):
    from hydra.utils import instantiate
    cfg = OmegaConf.load(config_path)
    if device.type == "cpu":
        cfg.model.attention_backend = "math"

    model = instantiate(cfg.model)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["model"]
    if any(k.startswith("net.module.") for k in state):
        state = {k.replace("net.module.", "net.", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"Loaded model from {checkpoint_path}")
    return model, cfg

# ──────────────────────────────────────────────────────────────────────────────
# Representation extraction
# ──────────────────────────────────────────────────────────────────────────────

def _squeeze_leading(t):
    """Remove leading size-1 dims until shape is (n_tokens, channels)."""
    while t.dim() > 2 and t.shape[0] == 1:
        t = t.squeeze(0)
    return t


def extract(model, cfg, data_path, n_jets, device):
    # ── data ──────────────────────────────────────────────────────────────────
    ds = TopTaggingDataset()
    ds.load_data(
        data_path,
        mode="test",
        momentum_float64=bool(cfg.data.get("momentum_float64", True)),
    )
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    # ── storage ───────────────────────────────────────────────────────────────
    jet_reps_buf      = [[] for _ in range(N_LAYERS)]
    part_reps_buf     = [[] for _ in range(N_LAYERS)]

    jet_labels        = []
    jet_mass_l        = []
    jet_mult_l        = []
    jet_tau1_l        = []
    jet_tau2_l        = []
    jet_tau3_l        = []

    part_jet_idx_l    = []
    part_pt_l         = []
    part_eta_l        = []
    part_phi_l        = []
    part_E_l          = []
    part_quartile_l   = []
    part_dR_l         = []

    jet_count = 0
    t0 = time.time()

    for batch in loader:
        if jet_count >= n_jets:
            break

        ptr    = batch.ptr.cpu().numpy()
        n_this = min(len(ptr) - 1, n_jets - jet_count)

        for bi in range(n_this):
            start, end   = int(ptr[bi]), int(ptr[bi + 1])
            n_phys_raw   = end - start
            n_phys       = min(n_phys_raw, K_MAX)

            # raw 4-momenta (un-rescaled) for physics labels
            p4_full = batch.x[start:end].numpy()   # (n_phys_raw, 4)
            p4      = p4_full[:n_phys]              # truncated

            # ── physics labels ─────────────────────────────────────────────
            label       = int(batch.label[bi].item())
            mass        = jet_mass(p4_full)          # use all particles for mass
            mult        = n_phys_raw
            tau1, tau2, tau3 = nsubjettiness(p4)

            pt_p        = _pt(p4).astype(np.float32)
            eta_p       = _eta(p4).astype(np.float32)
            phi_p       = _phi(p4).astype(np.float32)
            E_p         = p4[:, 0].astype(np.float32)
            quartiles   = pt_rank_quartile(p4)
            dR_p        = dR_to_jet_axis(p4)

            jet_labels.append(label)
            jet_mass_l.append(mass)
            jet_mult_l.append(mult)
            jet_tau1_l.append(tau1)
            jet_tau2_l.append(tau2)
            jet_tau3_l.append(tau3)

            part_jet_idx_l.extend([jet_count] * n_phys)
            part_pt_l.extend(pt_p.tolist())
            part_eta_l.extend(eta_p.tolist())
            part_phi_l.extend(phi_p.tolist())
            part_E_l.extend(E_p.tolist())
            part_quartile_l.extend(quartiles.tolist())
            part_dR_l.extend(dR_p.tolist())

            # ── single-jet forward pass with hooks ────────────────────────
            captured = [None] * N_LAYERS

            def _make_hook(layer_idx):
                def _hook(module, inp, out):
                    # out = (h_mv, h_s)  for both linear_in and GATrBlock
                    captured[layer_idx] = out[1].detach().cpu()
                return _hook

            hooks = []
            hooks.append(model.net.linear_in.register_forward_hook(_make_hook(0)))
            for li, blk in enumerate(model.net.blocks):
                hooks.append(blk.register_forward_hook(_make_hook(li + 1)))

            with torch.no_grad():
                fourmomenta_t = torch.tensor(p4_full, dtype=torch.float32, device=device)
                scalars_t = torch.zeros(n_phys_raw, 0, dtype=torch.float32, device=device)
                ptr_t = torch.tensor([0, n_phys_raw], dtype=torch.long, device=device)
                emb = embed_tagging_data(fourmomenta_t, scalars_t, ptr_t, cfg.data)
                token_offset = int(emb["is_spurion"].sum().item()) + (
                    0 if model.aggregator is not None else 1
                )
                _ = model(emb)

            for h in hooks:
                h.remove()

            # ── store representations ──────────────────────────────────────
            for li in range(N_LAYERS):
                h_s = _squeeze_leading(captured[li])   # (n_tokens, s_ch)

                # jet-level: global token at index 0
                jet_reps_buf[li].append(h_s[0].numpy().astype(np.float32))

                # particle-level: physical particles
                phys = h_s[token_offset : token_offset + n_phys].numpy()
                part_reps_buf[li].append(phys.astype(np.float32))

            jet_count += 1

        if jet_count % 500 == 0 or jet_count == n_jets:
            elapsed = time.time() - t0
            print(f"  [{elapsed:6.0f}s]  {jet_count}/{n_jets} jets processed")

    print(f"\nDone. Total jets: {jet_count}")

    # ── assemble output arrays ─────────────────────────────────────────────
    jet_reps  = np.stack(
        [np.stack(jet_reps_buf[li]) for li in range(N_LAYERS)]
    )  # (N_LAYERS, n_jets, s_ch)

    part_reps = np.stack(
        [np.vstack(part_reps_buf[li]) for li in range(N_LAYERS)]
    )  # (N_LAYERS, total_particles, s_ch)

    tau1 = np.array(jet_tau1_l, dtype=np.float32)
    tau2 = np.array(jet_tau2_l, dtype=np.float32)
    tau3 = np.array(jet_tau3_l, dtype=np.float32)

    return {
        # representations
        "jet_reps"                : jet_reps,
        "particle_reps"           : part_reps,
        "particle_jet_idx"        : np.array(part_jet_idx_l,  dtype=np.int32),
        # jet-level labels
        "jet_labels"              : np.array(jet_labels,       dtype=np.int32),
        "jet_mass"                : np.array(jet_mass_l,       dtype=np.float32),
        "jet_multiplicity"        : np.array(jet_mult_l,       dtype=np.int32),
        "jet_tau1"                : tau1,
        "jet_tau2"                : tau2,
        "jet_tau3"                : tau3,
        "jet_tau21"               : tau2 / (tau1 + 1e-10),
        # particle-level labels
        "particle_pt"             : np.array(part_pt_l,        dtype=np.float32),
        "particle_eta"            : np.array(part_eta_l,       dtype=np.float32),
        "particle_phi"            : np.array(part_phi_l,       dtype=np.float32),
        "particle_E"              : np.array(part_E_l,         dtype=np.float32),
        "particle_pt_rank_quartile": np.array(part_quartile_l, dtype=np.int32),
        "particle_dR_to_axis"     : np.array(part_dR_l,        dtype=np.float32),
    }

# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="results/models/topt_GATr_7327_run0_it200000.pt")
    ap.add_argument("--config",     default="runs/topt/GATr_7327/config.yaml")
    ap.add_argument("--data-path",  default="data/toptagging_full.npz")
    ap.add_argument("--output",     default="results/probe_data_lgatr.npz")
    ap.add_argument("--n-jets",     type=int, default=10000)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, cfg = load_model(args.checkpoint, args.config, device)

    print(f"\nExtracting representations ({args.n_jets} jets, {N_LAYERS} layers)...")
    data = extract(model, cfg, args.data_path, args.n_jets, device)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)

    print(f"\nSaved → {out}")
    print("Shapes:")
    for k, v in data.items():
        print(f"  {k:35s}: {str(v.shape):25s}  {v.dtype}")


if __name__ == "__main__":
    main()
