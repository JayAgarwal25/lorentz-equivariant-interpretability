#!/usr/bin/env python
"""Compute the 832 Lorentz-invariant features per token from L-GATr's h_mv.

Features (16 channels): 16 grade-0 and 16 grade-4 components, 136 grade-1 and 136 grade-3 pair
inner products, 256 grade-1 x dualized grade-3 cross terms, 136 + 136 grade-2 scalar and
pseudoscalar invariants. Inner products use <X~Y>_0 with the Cl(1,3) metric (+,-,-,-).

    python src/lgatr/compute_mv_invariants.py --input results/probes/lgatr/seed1001/probe_data_lgatr_mv.npz \\
        --output results/probes/lgatr/seed1001/probe_data_lgatr_mv_invariants.npz
"""

import argparse
import sys
import time
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, ".")

from gatr.primitives.bilinear import geometric_product

# ──────────────────────────────────────────────────────────────────────────────
# Metric (inner product) factors for ⟨X̃Y⟩₀ in Cl(1,3) with signature (+−−−).
# Source: gatr/primitives/invariants.py  _INNER_PRODUCT_FACTORS
# IP[k] = reversion_sign(grade(k)) * (blade_k)²
# ──────────────────────────────────────────────────────────────────────────────
IP = np.array([1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1, 1, 1, 1,-1,-1], dtype=np.float32)

# grade-by-grade slice indices
SLICE_G0 = slice(0, 1)    # 1 blade
SLICE_G1 = slice(1, 5)    # 4 blades: e0, e1, e2, e3
SLICE_G2 = slice(5, 11)   # 6 blades: e01, e02, e03, e12, e13, e23
SLICE_G3 = slice(11, 15)  # 4 blades: e012, e013, e023, e123
SLICE_G4 = slice(15, 16)  # 1 blade:  e0123

N_CH = 16   # number of multivector channels in h_mv


def _build_dualization_matrix():
    """
    Build the (4, 4) matrix D such that ŵ[1:5] = D @ T[11:15].
    ŵ = T · I  where I = e0123 (right multiplication by pseudoscalar).
    D[a, b] = grade-1 component a of (blade_{11+b} * I).
    """
    I_mv = torch.zeros(16, dtype=torch.float32)
    I_mv[15] = 1.0
    D = np.zeros((4, 4), dtype=np.float32)
    for b in range(4):
        blade = torch.zeros(16, dtype=torch.float32)
        blade[11 + b] = 1.0
        result = geometric_product(blade, I_mv).numpy()
        D[:, b] = result[1:5]
    return D


def _build_grade4_projection():
    """
    Build the (6, 6) matrix G such that ⟨B_i B_j⟩₄ = B_i[5:11] @ G @ B_j[5:11].
    G[a, b] = grade-4 component of (blade_{5+a} * blade_{5+b}).
    """
    G = np.zeros((6, 6), dtype=np.float32)
    for a in range(6):
        for b in range(6):
            ba = torch.zeros(16, dtype=torch.float32); ba[5 + a] = 1.0
            bb = torch.zeros(16, dtype=torch.float32); bb[5 + b] = 1.0
            G[a, b] = geometric_product(ba, bb).numpy()[15]
    return G


def compute_invariants_batch(h_mv_f32: np.ndarray,
                             dual_mat: np.ndarray,
                             gp4_mat:  np.ndarray) -> np.ndarray:
    """
    Compute all 832 Lorentz-invariant features for a batch of tokens.

    Parameters
    ----------
    h_mv_f32 : (n, 16, 16) float32 — multivector representations
    dual_mat  : (4, 4) float32     — grade-3 → grade-1 dualization map
    gp4_mat   : (6, 6) float32     — grade-2 × grade-2 → grade-4 projection

    Returns
    -------
    (n, 832) float32 invariant features
    """
    n = h_mv_f32.shape[0]
    ii, jj = np.triu_indices(N_CH)  # 136 symmetric pairs

    feats = []

    # ── 1. Direct grade-0 scalars: (n, 16) ─────────────────────────────────
    feats.append(h_mv_f32[:, :, 0])

    # ── 2. Direct grade-4 pseudoscalars: (n, 16) ───────────────────────────
    feats.append(h_mv_f32[:, :, 15])

    # ── 3. Grade-1 pairs: ⟨Ṽ_i V_j⟩₀ → (n, 136) ─────────────────────────
    g1   = h_mv_f32[:, :, 1:5]                          # (n, 16, 4)
    g1w  = g1 * IP[1:5]                                  # (n, 16, 4) — weighted
    gram = np.einsum('nia, nja -> nij', g1w, g1)         # (n, 16, 16)
    feats.append(gram[:, ii, jj])                        # (n, 136)

    # ── 4. Grade-3 pairs: ⟨T̃_i T_j⟩₀ → (n, 136) ─────────────────────────
    g3   = h_mv_f32[:, :, 11:15]                         # (n, 16, 4)
    g3w  = g3 * IP[11:15]                                # (n, 16, 4) — weighted
    gram = np.einsum('nia, nja -> nij', g3w, g3)         # (n, 16, 16)
    feats.append(gram[:, ii, jj])                        # (n, 136)

    # ── 5. Cross-grade: v_i · ŵ_j → (n, 256) ──────────────────────────────
    # ŵ_j = dualization of grade-3 channel j to grade-1
    # ŵ[n, c, :] = dual_mat @ g3[n, c, :]  (a 4-component grade-1 vector)
    g3d  = np.einsum('ab, ncb -> nca', dual_mat, g3)     # (n, 16, 4) — dualized grade-1
    # v_i · ŵ_j = Minkowski inner product (grade-1 IP factors)
    cross = np.einsum('a, nia, nja -> nij', IP[1:5], g1, g3d)  # (n, 16, 16)
    feats.append(cross.reshape(n, N_CH * N_CH))          # (n, 256) — NOT symmetric

    # ── 6. Grade-2 scalar: ⟨B̃_i B_j⟩₀ → (n, 136) ─────────────────────────
    g2   = h_mv_f32[:, :, 5:11]                          # (n, 16, 6)
    g2w  = g2 * IP[5:11]                                 # (n, 16, 6) — weighted
    gram = np.einsum('nia, nja -> nij', g2w, g2)         # (n, 16, 16)
    feats.append(gram[:, ii, jj])                        # (n, 136)

    # ── 7. Grade-2 pseudoscalar: ⟨B_i B_j⟩₄ → (n, 136) ───────────────────
    gram = np.einsum('ab, nia, njb -> nij', gp4_mat, g2, g2)  # (n, 16, 16)
    feats.append(gram[:, ii, jj])                              # (n, 136)

    return np.concatenate(feats, axis=1).astype(np.float32)   # (n, 832)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default="results/probe_data_lgatr_mv.npz")
    ap.add_argument("--output", default="results/probe_data_lgatr_mv_invariants.npz")
    ap.add_argument("--include-particles", dest="skip_particles", action="store_false",
                    help="Also compute particle-level invariants (about 18 GB); skipped by default.")
    args = ap.parse_args()

    t0 = time.time()

    # ── precompute structural matrices ──────────────────────────────────────
    print("Building dualization and grade-4 projection matrices...")
    dual_mat = _build_dualization_matrix()
    gp4_mat  = _build_grade4_projection()

    print(f"Dualization matrix D (grade-3→grade-1 via T*I):\n{dual_mat}")
    print(f"Grade-4 projection (grade-2×grade-2, first 3x3):\n{gp4_mat[:3,:3]}")

    # ── load h_mv ───────────────────────────────────────────────────────────
    print(f"\nLoading {args.input}...")
    data = np.load(args.input)
    jet_mv        = data["jet_mv_reps"].astype(np.float32)   # (13, n_jets, 16, 16)
    jet_labels    = data["jet_labels"]
    particle_jet_idx = data["particle_jet_idx"]

    n_layers = jet_mv.shape[0]
    n_jets   = jet_mv.shape[1]
    print(f"  jet_mv_reps: {jet_mv.shape}")

    if not args.skip_particles:
        part_mv = data["particle_mv_reps"].astype(np.float32)
        n_parts = part_mv.shape[1]
        print(f"  particle_mv_reps: {part_mv.shape}")

    # ── compute invariants layer by layer ────────────────────────────────────
    print(f"\nComputing 832 invariants per token ({n_layers} layers)...")
    jet_inv = np.zeros((n_layers, n_jets, 832), dtype=np.float32)

    if not args.skip_particles:
        part_inv = np.zeros((n_layers, n_parts, 832), dtype=np.float32)

    for li in range(n_layers):
        t1 = time.time()
        jet_inv[li] = compute_invariants_batch(jet_mv[li], dual_mat, gp4_mat)
        if not args.skip_particles:
            part_inv[li] = compute_invariants_batch(part_mv[li], dual_mat, gp4_mat)
        print(f"  layer {li:2d}  [{time.time()-t1:.1f}s]")

    # ── save ────────────────────────────────────────────────────────────────
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_dict = dict(
        jet_inv          = jet_inv,
        jet_labels       = jet_labels,
        particle_jet_idx = particle_jet_idx,
        dual_mat         = dual_mat,
        gp4_mat          = gp4_mat,
    )
    if not args.skip_particles:
        save_dict["particle_inv"] = part_inv
    np.savez_compressed(out, **save_dict)

    print(f"\nSaved → {out}  [{time.time()-t0:.0f}s total]")
    print(f"  jet_inv: {jet_inv.shape}   {jet_inv.nbytes/1e9:.2f} GB")
    if not args.skip_particles:
        print(f"  particle_inv: {part_inv.shape}   {part_inv.nbytes/1e9:.2f} GB")


if __name__ == "__main__":
    main()
