#!/usr/bin/env python
"""Extract per-layer multivector representations h_mv of L-GATr (16 channels x 16 components) for invariant probes.

Same hooks and token layout as run_probes_lgatr.py (particles start at token 4). Physics labels
are not saved here; they come from probe_data_lgatr.npz.

    python src/lgatr/run_probes_lgatr_mv.py --checkpoint $RUNS/topt_lgatr/seed1001/models/model_run0.pt \\
        --config $RUNS/topt_lgatr/seed1001/config.yaml --data-path $DATA --n-jets 10000 \\
        --output results/probes/lgatr/seed1001/probe_data_lgatr_mv.npz
"""

import argparse
import sys
import time
import numpy as np
import torch
from pathlib import Path
from omegaconf import OmegaConf
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

sys.path.insert(0, ".")

from experiments.tagging.embedding import embed_tagging_data
from experiments.tagging.dataset import TopTaggingDataset

# ──────────────────────────────────────────────────────────────────────────────

TOKEN_OFFSET   = 4    # global(0), beam+z(1), beam-z(2), time-spur(3), then particles
N_LAYERS       = 13   # 0 = after linear_in,  1..12 = after each GATrBlock
K_MAX          = 50   # max particles per jet to store
MV_CH          = 16   # hidden multivector channels
BLADE_DIM      = 16   # Cl(1,3) blade components

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


def _squeeze_mv(t):
    """Remove leading size-1 dims until shape is (n_tokens, ch, 16)."""
    while t.dim() > 3 and t.shape[0] == 1:
        t = t.squeeze(0)
    return t


def extract(model, cfg, data_path, n_jets, device):
    ds = TopTaggingDataset()
    ds.load_data(data_path, mode="test")
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    jet_mv_buf   = [[] for _ in range(N_LAYERS)]
    part_mv_buf  = [[] for _ in range(N_LAYERS)]

    jet_labels      = []
    part_jet_idx_l  = []

    jet_count = 0
    t0 = time.time()

    for batch in loader:
        if jet_count >= n_jets:
            break

        ptr    = batch.ptr.cpu().numpy()
        n_this = min(len(ptr) - 1, n_jets - jet_count)

        for bi in range(n_this):
            start, end    = int(ptr[bi]), int(ptr[bi + 1])
            n_phys_raw    = end - start
            n_phys        = min(n_phys_raw, K_MAX)

            label = int(batch.label[bi].item())
            jet_labels.append(label)
            part_jet_idx_l.extend([jet_count] * n_phys)

            # ── single-jet forward pass ─────────────────────────────────────
            scalars_in = (
                batch.scalars[start:end].clone()
                if hasattr(batch, "scalars") and batch.scalars is not None
                else None
            )
            sj = Data(
                x       = batch.x[start:end].clone(),
                scalars = scalars_in,
                ptr     = torch.tensor([0, n_phys_raw], dtype=torch.long),
                label   = batch.label[bi : bi + 1].clone(),
            )

            captured_mv = [None] * N_LAYERS

            def _make_hook(layer_idx):
                def _hook(module, inp, out):
                    # out = (h_mv, h_s) — capture h_mv = out[0]
                    captured_mv[layer_idx] = out[0].detach().cpu()
                return _hook

            hooks = []
            hooks.append(model.net.linear_in.register_forward_hook(_make_hook(0)))
            for li, blk in enumerate(model.net.blocks):
                hooks.append(blk.register_forward_hook(_make_hook(li + 1)))

            with torch.no_grad():
                sj_dev = sj.to(device)
                scalars_dev = (
                    sj_dev.scalars
                    if sj_dev.scalars is not None
                    else torch.zeros(n_phys_raw, 0, device=device)
                )
                emb = embed_tagging_data(
                    sj_dev.x, scalars_dev, sj_dev.ptr, cfg.data
                )
                _ = model(emb)

            for h in hooks:
                h.remove()

            # ── store representations ───────────────────────────────────────
            for li in range(N_LAYERS):
                h_mv = _squeeze_mv(captured_mv[li])  # (n_tokens, 16, 16)

                # jet-level: global token at index 0
                jet_mv_buf[li].append(
                    h_mv[0].numpy().astype(np.float16)  # (16, 16)
                )

                # particle-level: physical particles
                phys_mv = h_mv[TOKEN_OFFSET : TOKEN_OFFSET + n_phys].numpy()
                part_mv_buf[li].append(phys_mv.astype(np.float16))  # (n_phys, 16, 16)

            jet_count += 1

        if jet_count % 500 == 0 or jet_count == n_jets:
            print(f"  [{time.time()-t0:6.0f}s]  {jet_count}/{n_jets} jets processed")

    print(f"\nDone. Total jets: {jet_count}")

    # ── assemble ────────────────────────────────────────────────────────────
    # jet_mv_reps: (N_LAYERS, n_jets, 16, 16)
    jet_mv_reps = np.stack(
        [np.stack(jet_mv_buf[li]) for li in range(N_LAYERS)]
    ).astype(np.float16)

    # particle_mv_reps: (N_LAYERS, total_particles, 16, 16)
    particle_mv_reps = np.stack(
        [np.vstack(part_mv_buf[li]) for li in range(N_LAYERS)]
    ).astype(np.float16)

    return {
        "jet_mv_reps"      : jet_mv_reps,
        "particle_mv_reps" : particle_mv_reps,
        "particle_jet_idx" : np.array(part_jet_idx_l, dtype=np.int32),
        "jet_labels"       : np.array(jet_labels,     dtype=np.int32),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config",     required=True)
    ap.add_argument("--data-path",  default="data/toptagging_full.npz")
    ap.add_argument("--output",     default="results/probe_data_lgatr_mv.npz")
    ap.add_argument("--n-jets",     type=int, default=10000)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, cfg = load_model(args.checkpoint, args.config, device)
    print(f"\nExtracting h_mv ({args.n_jets} jets, {N_LAYERS} layers)...")
    data = extract(model, cfg, args.data_path, args.n_jets, device)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)

    print(f"\nSaved → {out}")
    for k, v in data.items():
        print(f"  {k:30s}: {str(v.shape):35s}  {v.dtype}")


if __name__ == "__main__":
    main()
