#!/usr/bin/env python3
"""LLoCa-specific Lorentz equivariance diagnostic for TopTagging.

This script tests the canonicalization architecture in a way that is distinct
from the vanilla Transformer baseline.  The default mode applies the same
Lorentz transform to the full physical event, including beam/time reference
spurions, before the model's own preprocessing (`boost_jet`, mass regulator,
and tagging-feature construction).  This avoids the main pitfall of the old
vanilla-style script, which transformed only the particle four-momenta and then
recreated fixed lab-frame spurions.

Saved arrays keep the existing equivariance schema so the shared bootstrap
script can be used unchanged:
  err_all     : (jets, transforms, layers)
  logit_errs  : (jets, transforms)
  sweep_err   : (gamma, jets, transforms)
  sweep_logit : (gamma, jets, transforms)
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict
from torch_geometric.utils import scatter

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")

from experiments.tagging.embedding import get_spurion, get_tagging_features
from lloca.utils.lorentz import lorentz_squarednorm
from lloca.utils.polar_decomposition import restframe_boost
from lloca.utils.utils import get_batch_from_ptr


EPS = 1e-8


def boost_matrix(gamma, direction=None, rng=None, dtype=np.float64):
    if rng is None:
        rng = np.random.default_rng()
    if direction is None:
        n = rng.standard_normal(3)
        n = n / np.linalg.norm(n)
    else:
        n = np.asarray(direction, dtype=dtype)
        n = n / np.linalg.norm(n)
    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    gb = gamma * beta
    gm1 = gamma - 1.0
    nx, ny, nz = n
    return np.array(
        [
            [gamma, gb * nx, gb * ny, gb * nz],
            [gb * nx, 1 + gm1 * nx * nx, gm1 * nx * ny, gm1 * nx * nz],
            [gb * ny, gm1 * nx * ny, 1 + gm1 * ny * ny, gm1 * ny * nz],
            [gb * nz, gm1 * nx * nz, gm1 * ny * nz, 1 + gm1 * nz * nz],
        ],
        dtype=dtype,
    )


def rotation_matrix(rng=None, dtype=np.float64):
    if rng is None:
        rng = np.random.default_rng()
    m = rng.standard_normal((3, 3))
    q, r = np.linalg.qr(m)
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    lam = np.eye(4, dtype=dtype)
    lam[1:4, 1:4] = q.astype(dtype)
    return lam


def sample_lorentz_transform(rng=None):
    if rng is None:
        rng = np.random.default_rng()
    gamma = float(np.exp(rng.uniform(np.log(1.5), np.log(3.0))))
    return (boost_matrix(gamma, rng=rng) @ rotation_matrix(rng)).astype(np.float64)


def sample_pure_boost(gamma, rng=None):
    return boost_matrix(float(gamma), rng=rng).astype(np.float64)


def rel_err(a, b, eps=EPS):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + eps))


def load_model(config_path, checkpoint_path, device):
    cfg = OmegaConf.load(config_path)
    with open_dict(cfg):
        if "compile" in cfg.model.net:
            cfg.model.net.compile = False
        if "compile" in cfg.model.framesnet:
            cfg.model.framesnet.compile = False
    model = instantiate(cfg.model)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    print(f"Loaded model from {checkpoint_path}")
    return model, cfg


def _apply_lambda(fourmomenta, lambda_t):
    return fourmomenta @ lambda_t


def embed_lloca_jet(p4_np, cfg_data, device, lambda_np=None, mode="full_event"):
    """Build an LLoCa embedding for one jet.

    Modes:
      full_event: add spurions, transform particles + spurions, then run the
        normal LLoCa data preprocessing. This is the architecture-specific
        equivariance test.
      physical_only: transform only particle four-momenta before embedding,
        recreating fixed spurions. This reproduces the old vanilla-style test.
      none: no transform.
    """
    momentum_dtype = torch.float64 if bool(cfg_data.momentum_float64) else torch.float32
    scalar_dtype = torch.float64 if bool(getattr(cfg_data, "network_float64", False)) else torch.float32

    fourmomenta = torch.tensor(p4_np, dtype=momentum_dtype, device=device)
    scalars = torch.zeros(fourmomenta.shape[0], 0, dtype=scalar_dtype, device=device)
    ptr = torch.tensor([0, fourmomenta.shape[0]], dtype=torch.long, device=device)

    lambda_t = None
    if lambda_np is not None:
        lambda_t = torch.tensor(lambda_np, dtype=momentum_dtype, device=device)
        if mode == "physical_only":
            fourmomenta = _apply_lambda(fourmomenta, lambda_t)

    batchsize = len(ptr) - 1
    arange = torch.arange(batchsize, device=fourmomenta.device)

    if cfg_data.max_particles is not None:
        counts = ptr[1:] - ptr[:-1]
        kept_counts = counts.clamp_max(cfg_data.max_particles)
        if kept_counts.max() > 0:
            kept_idx = torch.repeat_interleave(
                torch.arange(len(kept_counts), device=ptr.device), kept_counts
            )
            segment_start = torch.cat([kept_counts.new_zeros(1), kept_counts.cumsum(dim=0)[:-1]])
            start_offsets = torch.repeat_interleave(segment_start, kept_counts)
            local_idx = torch.arange(kept_counts.sum(), device=ptr.device) - start_offsets
            offsets = ptr[kept_idx] + local_idx
            fourmomenta = fourmomenta.index_select(0, offsets)
            scalars = scalars.index_select(0, offsets)
            ptr = torch.cat([ptr.new_zeros(1), kept_counts.cumsum(0)])

    spurions = get_spurion(
        cfg_data.beam_reference,
        cfg_data.add_time_reference,
        cfg_data.two_beams,
        fourmomenta.device,
        fourmomenta.dtype,
    )
    spurions *= cfg_data.spurion_scale

    n_spurions = spurions.shape[0]
    is_spurion = torch.zeros(
        fourmomenta.shape[0] + n_spurions * batchsize,
        dtype=torch.bool,
        device=fourmomenta.device,
    )
    if n_spurions > 0:
        spurion_idxs = torch.stack(
            [ptr[:-1] + i for i in range(n_spurions)], dim=0
        ) + n_spurions * arange
        spurion_idxs = spurion_idxs.permute(1, 0).flatten()
        is_spurion[spurion_idxs] = True

        fourmomenta_buffer = fourmomenta.clone()
        fourmomenta = torch.empty(
            is_spurion.shape[0],
            *fourmomenta.shape[1:],
            dtype=fourmomenta.dtype,
            device=fourmomenta.device,
        )
        fourmomenta[~is_spurion] = fourmomenta_buffer
        fourmomenta[is_spurion] = spurions.repeat(batchsize, 1)

        scalars_buffer = scalars.clone()
        scalars = torch.zeros(
            fourmomenta.shape[0],
            scalars.shape[1],
            dtype=scalars.dtype,
            device=scalars.device,
        )
        scalars[~is_spurion] = scalars_buffer
        ptr[1:] = ptr[1:] + (arange + 1) * n_spurions

    if lambda_t is not None and mode == "full_event":
        fourmomenta = _apply_lambda(fourmomenta, lambda_t)

    if cfg_data.mass_reg is not None:
        mass_reg = cfg_data.mass_reg
        mask = lorentz_squarednorm(fourmomenta) < mass_reg**2
        fourmomenta = fourmomenta.clone()
        fourmomenta[mask, 0] = (fourmomenta[mask, 1:] ** 2).sum(dim=-1).add(mass_reg**2).sqrt()

    batch = get_batch_from_ptr(ptr)

    if cfg_data.boost_jet:
        jet = scatter(
            fourmomenta[~is_spurion],
            batch[~is_spurion],
            dim=0,
            reduce="sum",
        ).index_select(0, batch)
        jet_boost = restframe_boost(jet)
        fourmomenta = torch.einsum("ijk,ik->ij", jet_boost, fourmomenta)

    jet = scatter(
        fourmomenta[~is_spurion],
        batch[~is_spurion],
        dim=0,
        reduce="sum",
    ).index_select(0, batch)
    tagging_features = get_tagging_features(
        fourmomenta,
        jet,
        tagging_features=cfg_data.tagging_features,
    )
    tagging_features[is_spurion] = 0
    tagging_features = tagging_features.to(scalars.dtype)

    return {
        "fourmomenta": fourmomenta,
        "scalars": scalars,
        "is_spurion": is_spurion,
        "tagging_features": tagging_features,
        "batch": batch,
        "ptr": ptr,
        "num_graphs": batchsize,
    }


def run_model(model, embedding):
    n_layers = 1 + len(model.net.blocks)
    hs_raw = [None] * n_layers

    def _make_hook(li):
        def hook(module, inp, out):
            hs_raw[li] = out.detach().cpu()

        return hook

    handles = [model.net.linear_in.register_forward_hook(_make_hook(0))]
    for li, block in enumerate(model.net.blocks):
        handles.append(block.register_forward_hook(_make_hook(li + 1)))

    with torch.no_grad():
        out = model(embedding)

    for handle in handles:
        handle.remove()

    hs_list = []
    for t in hs_raw:
        if t is None:
            hs_list.append(None)
            continue
        while t.ndim > 2 and t.shape[0] == 1:
            t = t.squeeze(0)
        hs_list.append(t.mean(dim=0).numpy().astype(np.float32))

    logits = out[0]
    logit = float(logits.squeeze()[1].cpu()) if logits.shape[-1] > 1 else float(logits.squeeze().cpu())
    return hs_list, logit


def iter_jets(npz, n_jets):
    kinematics = npz["kinematics_test"]
    jets = []
    for raw in kinematics:
        if len(jets) >= n_jets:
            break
        is_real = np.any(raw != 0, axis=1)
        if is_real.sum() < 3:
            continue
        idx = np.where(is_real)[0]
        jets.append(raw[idx].astype(np.float64))
    return jets


def test_equivariance(model, cfg, jets, n_transforms, device, rng, mode):
    n_layers = 1 + len(model.net.blocks)
    err_all = np.zeros((len(jets), n_transforms, n_layers), dtype=np.float32)
    logit_errs = np.zeros((len(jets), n_transforms), dtype=np.float32)
    transforms = [sample_lorentz_transform(rng) for _ in range(n_transforms)]

    t0 = time.time()
    for ji, p4 in enumerate(jets):
        emb_orig = embed_lloca_jet(p4, cfg.data, device, mode="none")
        hs_orig, logit_orig = run_model(model, emb_orig)
        for ti, lam in enumerate(transforms):
            emb_trans = embed_lloca_jet(p4, cfg.data, device, lambda_np=lam, mode=mode)
            hs_trans, logit_trans = run_model(model, emb_trans)
            logit_errs[ji, ti] = abs(logit_trans - logit_orig) / (abs(logit_orig) + EPS)
            for li in range(n_layers):
                if hs_orig[li] is not None and hs_trans[li] is not None:
                    err_all[ji, ti, li] = rel_err(hs_trans[li], hs_orig[li])
        if (ji + 1) % 50 == 0:
            print(f"  LLoCa {mode}: {ji + 1}/{len(jets)} jets [{time.time() - t0:.0f}s]")

    return err_all, logit_errs


def test_boost_sweep(model, cfg, jets, gammas, n_transforms, device, rng, mode):
    n_g = len(gammas)
    n_layers = 1 + len(model.net.blocks)
    sweep_err = np.zeros((n_g, len(jets), n_transforms), dtype=np.float32)
    sweep_logit = np.zeros((n_g, len(jets), n_transforms), dtype=np.float32)

    t0 = time.time()
    for gi, gamma in enumerate(gammas):
        transforms = [sample_pure_boost(gamma, rng) for _ in range(n_transforms)]
        for ji, p4 in enumerate(jets):
            emb_orig = embed_lloca_jet(p4, cfg.data, device, mode="none")
            hs_orig, logit_orig = run_model(model, emb_orig)
            for ti, lam in enumerate(transforms):
                emb_trans = embed_lloca_jet(p4, cfg.data, device, lambda_np=lam, mode=mode)
                hs_trans, logit_trans = run_model(model, emb_trans)
                sweep_logit[gi, ji, ti] = abs(logit_trans - logit_orig) / (abs(logit_orig) + EPS)
                layer_errs = [
                    rel_err(hs_trans[li], hs_orig[li])
                    for li in range(n_layers)
                    if hs_orig[li] is not None and hs_trans[li] is not None
                ]
                sweep_err[gi, ji, ti] = float(np.mean(layer_errs))
        print(
            f"  gamma={gamma:5.1f}: hidden={np.nanmean(sweep_err[gi]):.3e} "
            f"logit={np.nanmean(sweep_logit[gi]):.3e} [{time.time() - t0:.0f}s]"
        )
    return sweep_err, sweep_logit


def plot_boost_sweep(gammas, sweep_err, sweep_logit, save_path, mode):
    mean_err = np.nanmean(sweep_err, axis=(1, 2))
    mean_logit = np.nanmean(sweep_logit, axis=(1, 2))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(gammas, mean_err, "o-", linewidth=2, label="hidden representation")
    ax.loglog(gammas, mean_logit, "s-", linewidth=2, label="output logit")
    ax.set_xlabel("Lorentz boost factor gamma")
    ax.set_ylabel("Mean relative change")
    ax.set_title(f"LLoCa-T equivariance sweep ({mode})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_summary(path, err_all, logit_errs, gammas, sweep_err, sweep_logit, mode):
    lines = [
        "LLoCa-T Lorentz Equivariance Diagnostic",
        f"mode={mode}",
        f"n_jets={logit_errs.shape[0]} n_transforms={logit_errs.shape[1]}",
        "",
        "Random Lorentz transforms:",
        f"  hidden mean: {np.nanmean(err_all):.6g}",
        f"  logit mean:  {np.nanmean(logit_errs):.6g}",
        f"  logit median:{np.nanmedian(logit_errs):.6g}",
        "",
        "Boost sweep:",
        f"  {'gamma':>8} {'hidden mean':>14} {'logit mean':>14} {'nonfinite logit':>16}",
    ]
    for gi, gamma in enumerate(gammas):
        nonfinite = int((~np.isfinite(sweep_logit[gi])).sum())
        lines.append(
            f"  {gamma:8.1f} {np.nanmean(sweep_err[gi]):14.6g} "
            f"{np.nanmean(sweep_logit[gi]):14.6g} {nonfinite:16d}"
        )
    text = "\n".join(lines)
    path.write_text(text)
    print(text)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-path", default="data/toptagging_full.npz")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--mode", choices=["full_event", "physical_only"], default="full_event")
    ap.add_argument("--n-jets", type=int, default=200)
    ap.add_argument("--n-transforms", type=int, default=5)
    ap.add_argument("--n-jets-sweep", type=int, default=50)
    ap.add_argument("--gammas", type=float, nargs="+", default=[1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0])
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    print(f"Device: {device}")
    print(f"mode={args.mode} seed={args.seed}")

    model, cfg = load_model(args.config, args.checkpoint, device)
    npz = np.load(args.data_path)
    jets = iter_jets(npz, args.n_jets)
    sweep_jets = jets[: args.n_jets_sweep]

    err_all, logit_errs = test_equivariance(
        model, cfg, jets, args.n_transforms, device, rng, args.mode
    )
    sweep_err, sweep_logit = test_boost_sweep(
        model, cfg, sweep_jets, args.gammas, args.n_transforms, device, rng, args.mode
    )

    np.savez_compressed(
        out_dir / "equivariance_lloca_topt.npz",
        err_all=err_all,
        logit_errs=logit_errs,
        gammas=np.asarray(args.gammas, dtype=np.float32),
        sweep_err=sweep_err,
        sweep_logit=sweep_logit,
    )
    plot_boost_sweep(
        np.asarray(args.gammas, dtype=np.float32),
        sweep_err,
        sweep_logit,
        out_dir / "equivariance_lloca_boost_sweep.png",
        args.mode,
    )
    write_summary(
        out_dir / "equivariance_lloca_summary.txt",
        err_all,
        logit_errs,
        np.asarray(args.gammas, dtype=np.float32),
        sweep_err,
        sweep_logit,
        args.mode,
    )
    print(f"Saved outputs in {out_dir}")


if __name__ == "__main__":
    main()
