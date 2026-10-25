#!/usr/bin/env python3
"""Dedicated TopTagging probe extraction for ParT.

This is the probe-only counterpart to ``run_interp_part.py``. It uses the
wrapper-era ``lloca-experiments`` ParT checkpoint/config path, extracts:

- particle-token representations after embed + each particle block
- jet-level representations from:
  - mean-pooled particle tokens after embed + each particle block
  - class-token states after each cls block
  - final normalized class-token state before the FC head

The final normalized class-token state is the representation closest to the
actual classifier logits, so this script is the right extraction path for the
TopTagging linear probes.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from run_interp_part import (
    Data,
    DataLoader,
    TopTaggingDataset,
    _eta,
    _phi,
    _pt,
    dR_to_jet_axis,
    embed_tagging_data,
    jet_mass,
    load_model,
    nsubjettiness,
    pt_rank_quartile,
)


def _squeeze_token_state(tensor):
    while tensor.dim() > 2 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    return tensor


def _get_core_part_model(model):
    core = getattr(model, "net", model)
    return getattr(core, "mod", core)


def extract_probe_data(model, cfg, data_path, n_jets, k_store, device, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    momentum_float64 = bool(cfg.data.get("momentum_float64", True))
    ds = TopTaggingDataset()
    ds.load_data(data_path, mode="test", momentum_float64=momentum_float64)
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    core_model = _get_core_part_model(model)

    particle_modules = [("embed", core_model.embed)]
    particle_modules.extend((f"block{i + 1}", blk) for i, blk in enumerate(core_model.blocks))
    cls_modules = [(f"cls{i + 1}", blk) for i, blk in enumerate(core_model.cls_blocks)]
    cls_modules.append(("norm", core_model.norm))

    n_part_layers = len(particle_modules)
    n_jet_layers = n_part_layers + len(cls_modules)

    particle_captured = [None] * n_part_layers
    cls_captured = [None] * len(cls_modules)

    def _make_particle_hook(idx):
        def hook(module, inp, out):
            particle_captured[idx] = out.detach().cpu()
        return hook

    def _make_cls_hook(idx):
        def hook(module, inp, out):
            cls_captured[idx] = out.detach().cpu()
        return hook

    hooks = []
    for idx, (_, module) in enumerate(particle_modules):
        hooks.append(module.register_forward_hook(_make_particle_hook(idx)))
    for idx, (_, module) in enumerate(cls_modules):
        hooks.append(module.register_forward_hook(_make_cls_hook(idx)))

    jet_reps_buf = [[] for _ in range(n_jet_layers)]
    part_reps_buf = [[] for _ in range(n_part_layers)]

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
            if jet_count >= n_jets:
                break

            ptr = batch.ptr.cpu().numpy()
            n_this = min(len(ptr) - 1, n_jets - jet_count)

            for bi in range(n_this):
                start, end = int(ptr[bi]), int(ptr[bi + 1])
                n_phys_raw = end - start
                if n_phys_raw < 3:
                    continue

                n_use = min(n_phys_raw, k_store)
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

                for idx in range(n_part_layers):
                    particle_captured[idx] = None
                for idx in range(len(cls_modules)):
                    cls_captured[idx] = None

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
                mass = jet_mass(p4_full)
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

                last_part_rep = None
                for li in range(n_part_layers):
                    h = particle_captured[li]
                    if h is None:
                        raise RuntimeError(f"Missing particle hook output for layer {li}")
                    h_np = _squeeze_token_state(h).numpy().astype(np.float32)
                    jet_rep = h_np.mean(axis=0)
                    part_rep = h_np[:n_use]
                    jet_reps_buf[li].append(jet_rep)
                    part_reps_buf[li].append(part_rep)
                    last_part_rep = part_rep

                if last_part_rep is None:
                    raise RuntimeError("No particle representations were captured")

                for ci, captured in enumerate(cls_captured):
                    if captured is None:
                        raise RuntimeError(f"Missing cls/norm hook output for stage {ci}")
                    cls_np = _squeeze_token_state(captured).numpy().astype(np.float32)
                    if cls_np.ndim == 2:
                        cls_rep = cls_np[0]
                    else:
                        cls_rep = cls_np
                    jet_reps_buf[n_part_layers + ci].append(cls_rep)

                jet_count += 1
                if jet_count % 500 == 0:
                    elapsed = time.time() - t0
                    print(f"  [{elapsed:5.0f}s]  {jet_count}/{n_jets} jets")

    finally:
        for hook in hooks:
            hook.remove()

    if jet_count == 0:
        raise RuntimeError("No valid jets were processed")

    jet_reps = np.stack([np.stack(jet_reps_buf[li]) for li in range(n_jet_layers)])
    particle_layers = [np.vstack(part_reps_buf[li]) for li in range(n_part_layers)]
    while len(particle_layers) < n_jet_layers:
        particle_layers.append(particle_layers[-1].copy())
    particle_reps = np.stack(particle_layers)

    tau1 = np.array(jet_tau1_l, dtype=np.float32)
    tau2 = np.array(jet_tau2_l, dtype=np.float32)
    tau3 = np.array(jet_tau3_l, dtype=np.float32)

    probe_data = {
        "jet_reps": jet_reps,
        "particle_reps": particle_reps,
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
        "jet_layer_names": np.array(
            [name for name, _ in particle_modules] + [name for name, _ in cls_modules]
        ),
        "particle_layer_names": np.array(
            [name for name, _ in particle_modules]
            + [particle_modules[-1][0]] * (n_jet_layers - n_part_layers)
        ),
        "n_particle_layers": np.array(n_part_layers, dtype=np.int32),
        "n_jet_layers": np.array(n_jet_layers, dtype=np.int32),
    }

    np.savez_compressed(output, **probe_data)
    print(f"Saved probe data → {output}")
    for key, val in probe_data.items():
        shape = getattr(val, "shape", ())
        dtype = getattr(val, "dtype", type(val).__name__)
        print(f"  {key:35s}: {str(shape):25s}  {dtype}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-jets", type=int, default=10000)
    ap.add_argument("--k-store", type=int, default=50)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model, cfg, use_wrapper = load_model(args.checkpoint, device, config_path=args.config)
    if not use_wrapper or cfg is None:
        raise RuntimeError("run_probes_part_topt.py requires a wrapper-era config/checkpoint pair")

    extract_probe_data(
        model,
        cfg,
        data_path=args.data_path,
        n_jets=args.n_jets,
        k_store=args.k_store,
        device=device,
        output=args.output,
    )


if __name__ == "__main__":
    main()
