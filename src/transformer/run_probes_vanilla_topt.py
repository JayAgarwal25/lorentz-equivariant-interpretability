#!/usr/bin/env python3
"""
Extract per-layer representations from Transformer-style TopTagging models for linear probing.

Architecture:
  TransformerWrapper → lloca.backbone.transformer.Transformer
  linear_in + all model.net.blocks → hook points L0..L{n_blocks}
  mean_aggregation=True → no global token; jet rep = mean pool over physical particles
  Spurions are stripped before model.net — model sees ONLY physical particles, no spurions
  Block output: plain tensor (n_particles, hidden_dim), NOT a (h_mv, h_s) tuple

Hook targets:
  Layer 0: model.net.linear_in  → output tensor (n_tokens, hidden_dim)
  Layer 1-10: model.net.blocks[i] → output tensor (n_tokens, hidden_dim)

Jet-level rep: mean over particle tokens [ptr[i]:ptr[i+1]] (no global token)
Particle-level rep: each physical particle token directly (no offset needed)

Usage (from lloca-experiments root):
  CUDA_VISIBLE_DEVICES=0 nohup python3.10 run_probes_vanilla_topt.py \\
      --config runs/vanilla_transformer_topt/Transformer_3909/config_0.yaml \\
      --checkpoint runs/vanilla_transformer_topt/Transformer_3909/models/model_run0.pt \\
      --data-path ../lorentz-gatr/data/toptagging_full.npz \\
      --output results/probe_data_vanilla_topt.npz \\
      --n-jets 10000 --truncate-particles 50 \\
      > results/probe_vanilla_topt.log 2>&1 &
"""

import argparse
import sys
import time
import numpy as np
import torch
from pathlib import Path
from omegaconf import OmegaConf, open_dict
from hydra.utils import instantiate
from torch_geometric.loader import DataLoader

sys.path.insert(0, ".")
from experiments.tagging.dataset import TopTaggingDataset
from experiments.tagging.embedding import embed_tagging_data

N_LAYERS = None # Filled from the checkpoint config; current LLoCa configs use 12 blocks.
K_MAX = 50      # max particles per jet to store
# TOKEN_OFFSET = 0: TransformerWrapper strips spurions before calling model.net,
# so h_batch is indexed by ptr_raw (physical particles only, no spurions in h_batch).


def load_model(config_path, checkpoint_path, device):
    cfg = OmegaConf.load(config_path)
    with open_dict(cfg):
        cfg.model.net.compile = False
    model = instantiate(cfg.model)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    print(f"Loaded model from {checkpoint_path}")
    return model, cfg


def extract(model, cfg, loader, n_jets, output_path, device):
    n_layers = 1 + len(model.net.blocks)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Storage
    jet_reps_buf  = [[] for _ in range(n_layers)]
    part_reps_buf = [[] for _ in range(n_layers)]

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

    # Hook storage: {layer_idx: tensor (n_tokens_batch, hidden_dim)}
    h_batch = {}

    def _make_hook(li):
        def hook(module, inp, out):
            # out shape: (batch=1, n_tokens, hidden_dim) for vanilla transformer
            # Convert to numpy and squeeze batch dim — avoids PyTorch CPU allocator VA accumulation
            h_np = out.detach().cpu().numpy()
            while h_np.ndim > 2 and h_np.shape[0] == 1:
                h_np = h_np.squeeze(0)
            h_batch[li] = h_np  # (n_tokens, hidden_dim)
        return hook

    handles = []
    handles.append(model.net.linear_in.register_forward_hook(_make_hook(0)))
    for li, blk in enumerate(model.net.blocks):
        handles.append(blk.register_forward_hook(_make_hook(li + 1)))

    jet_count = 0
    t0 = time.time()

    try:
        with torch.no_grad():
            for batch in loader:
                if jet_count >= n_jets:
                    break

                # Save raw ptr BEFORE embed_tagging_data (for p4 extraction from batch.x)
                ptr_raw = batch.ptr.clone().cpu()
                fourmomenta_cpu = batch.x.cpu().numpy()  # (n_particles_total, 4): E,px,py,pz
                labels_cpu = batch.label.cpu().numpy()

                batch_dev = batch.to(device)
                scalars = (
                    batch_dev.scalars
                    if (hasattr(batch_dev, "scalars") and batch_dev.scalars is not None)
                    else torch.zeros(batch_dev.x.shape[0], 0, device=device)
                )

                embedding = embed_tagging_data(
                    batch_dev.x, scalars, batch_dev.ptr, cfg.data
                )
                embedding["num_graphs"] = batch_dev.ptr.shape[0] - 1
                # ptr in embedding has been modified by embed_tagging_data (may add spurions/padding)
                ptr_net = embedding["ptr"].clone().cpu()

                h_batch.clear()
                model(embedding)

                B = len(labels_cpu)
                for i in range(B):
                    if jet_count >= n_jets:
                        break

                    raw_start = int(ptr_raw[i])
                    raw_end   = int(ptr_raw[i + 1])
                    n_phys    = raw_end - raw_start

                    if n_phys < 3:
                        continue

                    # h_batch is indexed by ptr_raw (spurions already stripped by wrapper)
                    part_start = raw_start
                    part_end   = raw_end
                    n_net      = n_phys

                    p4_full = fourmomenta_cpu[raw_start:raw_end]  # (n_phys, 4): E,px,py,pz
                    n_use   = min(n_phys, K_MAX)
                    p4      = p4_full[:n_use]

                    # Skip if any hook didn't fire — keeps all buffers aligned
                    if not all(h_batch.get(li) is not None for li in range(n_layers)):
                        continue

                    # Physics labels
                    label = int(labels_cpu[i])
                    mass  = _jet_mass(p4_full)
                    tau1, tau2, tau3 = _nsubjettiness(p4)
                    pt_p  = _pt(p4).astype(np.float32)
                    eta_p = _eta(p4).astype(np.float32)
                    phi_p = _phi(p4).astype(np.float32)
                    E_p   = p4[:, 0].astype(np.float32)
                    quart = _pt_rank_quartile(p4)
                    dR_p  = _dR_to_jet_axis(p4)

                    jet_labels_l.append(label)
                    jet_mass_l.append(mass)
                    jet_mult_l.append(n_phys)
                    jet_tau1_l.append(tau1)
                    jet_tau2_l.append(tau2)
                    jet_tau3_l.append(tau3)
                    part_jet_idx_l.extend([jet_count] * n_use)
                    part_pt_l.extend(pt_p.tolist())
                    part_eta_l.extend(eta_p.tolist())
                    part_phi_l.extend(phi_p.tolist())
                    part_E_l.extend(E_p.tolist())
                    part_quartile_l.extend(quart.tolist())
                    part_dR_l.extend(dR_p.tolist())

                    for li in range(n_layers):
                        h = h_batch[li]          # (total_phys_particles_in_batch, hidden_dim)
                        part_h = h[part_start:part_end, :]       # (n_phys, hidden_dim)
                        jet_h  = part_h.mean(axis=0).astype(np.float32)
                        part_h_use = part_h[:n_use].astype(np.float32)

                        jet_reps_buf[li].append(jet_h)
                        part_reps_buf[li].append(part_h_use)

                    jet_count += 1

                elapsed = time.time() - t0
                if jet_count % 500 < B:
                    print(f"  [{elapsed:5.0f}s]  {jet_count}/{n_jets} jets", flush=True)

    finally:
        for h in handles:
            h.remove()

    print(f"\nDone. Total jets: {jet_count}")

    # Check buffer consistency
    print(f"\nBuffer lengths per layer (jet_reps):")
    for li in range(n_layers):
        print(f"  L{li}: {len(jet_reps_buf[li])} jets")

    print(f"\nStacking...")
    jet_reps  = np.stack([np.stack(jet_reps_buf[li])   for li in range(n_layers)])
    part_reps = np.stack([np.vstack(part_reps_buf[li]) for li in range(n_layers)])

    tau1 = np.array(jet_tau1_l, dtype=np.float32)
    tau2 = np.array(jet_tau2_l, dtype=np.float32)
    tau3 = np.array(jet_tau3_l, dtype=np.float32)

    data = {
        "jet_reps"                 : jet_reps,
        "particle_reps"            : part_reps,
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
        "n_layers"                 : np.array(n_layers,         dtype=np.int32),
        "layer_names"              : np.array(["linear_in"] + [f"block_{i}" for i in range(n_layers - 1)]),
    }

    np.savez_compressed(output_path, **data)
    print(f"\nSaved → {output_path}")
    for k, v in data.items():
        print(f"  {k:35s}: {str(v.shape):25s}  {v.dtype}")
    return data


# ── Physics helpers (copied from run_probes_jetclass.py) ─────────────────────

def _pt(p4):
    return np.sqrt(p4[:, 1]**2 + p4[:, 2]**2)

def _eta(p4):
    pt = _pt(p4)
    return np.arcsinh(p4[:, 3] / (pt + 1e-10))

def _phi(p4):
    return np.arctan2(p4[:, 2], p4[:, 1])

def _jet_mass(p4):
    j  = p4.sum(axis=0)
    m2 = j[0]**2 - j[1]**2 - j[2]**2 - j[3]**2
    return float(np.sqrt(max(m2, 0.0)))

def _dR_to_jet_axis(p4):
    j     = p4.sum(axis=0)
    pt_j  = np.sqrt(j[1]**2 + j[2]**2)
    eta_j = float(np.arcsinh(j[3] / (pt_j + 1e-10)))
    phi_j = float(np.arctan2(j[2], j[1]))
    deta  = _eta(p4) - eta_j
    dphi  = (_phi(p4) - phi_j + np.pi) % (2 * np.pi) - np.pi
    return np.sqrt(deta**2 + dphi**2).astype(np.float32)

def _pt_rank_quartile(p4):
    n   = len(p4)
    pt  = _pt(p4)
    rnk = np.argsort(np.argsort(-pt))
    return (rnk * 4 // n).clip(0, 3).astype(np.int32)

def _exclusive_kt_merge(cp, ce, cphi, target_n):
    while len(cp) > target_n:
        n    = len(cp)
        pt_i = cp[:, None];  pt_j = cp[None, :]
        de   = ce[:, None] - ce[None, :]
        dp   = cphi[:, None] - cphi[None, :]
        dp   = (dp + np.pi) % (2 * np.pi) - np.pi
        d    = np.minimum(pt_i**2, pt_j**2) * (de**2 + dp**2)
        np.fill_diagonal(d, np.inf)
        flat = np.argmin(d); i, j = flat // n, flat % n
        if i > j: i, j = j, i
        pt_new  = cp[i] + cp[j]
        wj      = cp[j] / (pt_new + 1e-10)
        eta_new = ce[i] + wj * (ce[j] - ce[i])
        dpij    = (cphi[j] - cphi[i] + np.pi) % (2 * np.pi) - np.pi
        phi_new = cphi[i] + wj * dpij
        mask        = np.ones(n, dtype=bool)
        mask[i]     = False; mask[j] = False
        cp          = np.append(cp[mask],   pt_new)
        ce          = np.append(ce[mask],   eta_new)
        cphi        = np.append(cphi[mask], phi_new)
    return cp, ce, cphi

def _tau_from_axes(pt, eta, phi, axes, R=0.8):
    d0     = pt.sum() * R + 1e-10
    ax_eta = np.array([a[0] for a in axes])
    ax_phi = np.array([a[1] for a in axes])
    tau    = 0.0
    for i in range(len(pt)):
        de  = eta[i] - ax_eta
        dp  = (phi[i] - ax_phi + np.pi) % (2 * np.pi) - np.pi
        tau += pt[i] * np.sqrt(de**2 + dp**2).min()
    return tau / d0

def _nsubjettiness(p4, R=0.8):
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
    for n in (2, 3):
        if n not in axes_at:
            axes_at[n] = axes_at[n - 1]
    return (_tau_from_axes(pt, eta, phi, axes_at[1], R),
            _tau_from_axes(pt, eta, phi, axes_at[2], R),
            _tau_from_axes(pt, eta, phi, axes_at[3], R))


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",
        default="runs/vanilla_transformer_topt/Transformer_3909/config_0.yaml")
    ap.add_argument("--checkpoint",
        default="runs/vanilla_transformer_topt/Transformer_3909/models/model_run0.pt")
    ap.add_argument("--data-path",
        default="../lorentz-gatr/data/toptagging_full.npz")
    ap.add_argument("--output",
        default="results/probe_data_vanilla_topt.npz")
    ap.add_argument("--n-jets",      type=int, default=10000)
    ap.add_argument("--truncate-particles", type=int, default=50)
    ap.add_argument("--batch-size",  type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, cfg = load_model(args.config, args.checkpoint, device)

    # TopTagging dataset (same as lorentz-gatr probe scripts)
    ds = TopTaggingDataset()
    ds.load_data(args.data_path, mode="test")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    global K_MAX
    K_MAX = args.truncate_particles

    extract(model, cfg, loader, args.n_jets, args.output, device)


if __name__ == "__main__":
    main()
