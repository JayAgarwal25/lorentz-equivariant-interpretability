#!/usr/bin/env python3
"""Attention maps, pT-rank attention and layer CKA for LLoCa-T.

The backbone sees only the particles (the spurions are used for the local frames), so there is no
spurion attention. CKA uses the mean particle representation of --n-cka jets.

    python src/transformer/run_extended_interp_lloca_transformer.py --checkpoint $RUNS/topt_lloca_transformer/seed1001/models/model_run0.pt \\
        --config $RUNS/topt_lloca_transformer/seed1001/config.yaml --data-path $DATA --n-jets 10000 \\
        --batch-size 64 --output-dir results/interp/lloca_transformer/seed1001
"""

import argparse
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from omegaconf import OmegaConf
from torch_geometric.loader import DataLoader

plt.rcParams.update({"text.usetex": False, "font.family": "sans-serif"})

NUM_LAYERS = 13   # linear_in (L0) + 12 blocks (L1..L12)
NUM_HEADS  = 8
K          = 50   # max physical particles per jet


# ── CKA ──────────────────────────────────────────────────────────────────────

def _centered_gram(X):
    K_mat = X @ X.T
    rm = K_mat.mean(axis=1, keepdims=True)
    cm = K_mat.mean(axis=0, keepdims=True)
    gm = K_mat.mean()
    return K_mat - rm - cm + gm


def compute_cka_matrix(hs_accum):
    print(f"  Pre-computing centred Gram matrices ({NUM_LAYERS} layers)...")
    X  = [np.stack(hs_accum[i]).astype(np.float64) for i in range(NUM_LAYERS)]
    Kc = [_centered_gram(xi) for xi in X]
    diag = [np.sum(k * k) for k in Kc]
    cka = np.zeros((NUM_LAYERS, NUM_LAYERS))
    for i in range(NUM_LAYERS):
        for j in range(i, NUM_LAYERS):
            val = np.sum(Kc[i] * Kc[j]) / (np.sqrt(diag[i] * diag[j]) + 1e-10)
            cka[i, j] = cka[j, i] = val
    return cka


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(checkpoint_path, config_path, device):
    from hydra.utils import instantiate
    cfg = OmegaConf.load(config_path)

    if "compile" in cfg.model.net:
        del cfg.model.net["compile"]
    if device.type == "cpu":
        cfg.model.attention_backend = "math"

    model = instantiate(cfg.model)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["model"]
    if any(k.startswith("net.module.") for k in state):
        state = {k.replace("net.module.", "net.", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"Loaded model from {checkpoint_path}")
    print(f"  hidden_channels={model.net.hidden_channels}  num_heads={NUM_HEADS}")
    return model, cfg


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_cka(cka_matrix, title, save_path):
    layer_labels = ["L0"] + [f"L{i}" for i in range(1, NUM_LAYERS)]
    fig, ax = plt.subplots(figsize=(11, 10))
    sns.heatmap(
        cka_matrix, ax=ax, cmap="viridis", vmin=0, vmax=1,
        annot=True, fmt=".2f", annot_kws={"size": 6},
        xticklabels=layer_labels, yticklabels=layer_labels,
    )
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved CKA plot → {save_path}")


def plot_ptrank(top_list, qcd_list, title, save_path):
    def _mean_per_rank(arr_list):
        if not arr_list:
            return np.array([]), np.array([])
        max_k = max(len(a) for a in arr_list)
        s = np.zeros(max_k)
        c = np.zeros(max_k, dtype=int)
        for a in arr_list:
            s[:len(a)] += a
            c[:len(a)] += 1
        return np.arange(max_k), np.where(c > 0, s / c, 0.0)

    ranks_t, mean_t = _mean_per_rank(top_list)
    ranks_q, mean_q = _mean_per_rank(qcd_list)

    fig, ax = plt.subplots(figsize=(10, 5))
    if len(ranks_t):
        ax.plot(ranks_t, mean_t, label="Top", color="tab:blue",   linewidth=2)
    if len(ranks_q):
        ax.plot(ranks_q, mean_q, label="QCD", color="tab:orange", linewidth=2)
    ax.set_xlabel("pT rank (0 = highest pT)", fontsize=13)
    ax.set_ylabel("Mean attention weight received", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved pT-rank plot → {save_path}")


def plot_averaged_attention_grid(avg_attn, n_events, label="", title_prefix="LLoCa-Transformer", save_path=None):
    """One heatmap per attention head, averaged over n_events jets."""
    num_heads = avg_attn.shape[0]
    ncols = 4
    nrows = (num_heads + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]
    title = f"{title_prefix}  Averaged Attention Maps"
    if label:
        title += f"  ({label})"
    title += f"\nFinal Layer · {n_events} events"
    fig.suptitle(title, fontsize=16)
    for h in range(num_heads):
        ax = axes[h // ncols, h % ncols]
        sns.heatmap(avg_attn[h], ax=ax, cmap="viridis", cbar=True,
                    square=True, xticklabels=False, yticklabels=False)
        ax.set_title(f"Head {h}", fontsize=12)
    for h in range(num_heads, nrows * ncols):
        axes[h // ncols, h % ncols].set_visible(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved averaged attention grid → {save_path}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"n_jets={args.n_jets}  n_cka={args.n_cka}  K={K}  batch_size={args.batch_size}")

    model, cfg = load_model(args.checkpoint, args.config, device)

    from experiments.tagging.dataset import TopTaggingDataset
    from experiments.tagging.embedding import embed_tagging_data

    ds = TopTaggingDataset()
    ds.load_data(args.data_path, mode="test")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    # ── Accumulators ─────────────────────────────────────────────────────────
    hs_cka  = [[] for _ in range(NUM_LAYERS)]
    n_cka   = 0

    ptrank_top, ptrank_qcd = [], []

    attn_accum = {
        "Top": {"sum": np.zeros((NUM_HEADS, K, K), dtype=np.float64), "count": 0},
        "QCD": {"sum": np.zeros((NUM_HEADS, K, K), dtype=np.float64), "count": 0},
        "All": {"sum": np.zeros((NUM_HEADS, K, K), dtype=np.float64), "count": 0},
    }

    n_processed = 0
    t0 = time.time()

    with torch.no_grad():
        for batch in loader:
            if n_processed >= args.n_jets:
                break

            ptr    = batch.ptr.cpu().numpy()
            labels = batch.label.cpu().numpy()
            n_this = min(len(ptr) - 1, args.n_jets - n_processed)

            for bi in range(n_this):
                start, end = int(ptr[bi]), int(ptr[bi + 1])
                n_phys     = end - start

                if n_phys < 5:
                    n_processed += 1
                    continue

                k_eff = min(K, n_phys)

                # ── Single-jet embedding ──────────────────────────────────────
                fourmomenta_t = batch.x[start:end].to(device)
                scalars_t     = torch.zeros(n_phys, 0, device=device)
                ptr_t         = torch.tensor([0, n_phys], dtype=torch.long, device=device)
                emb = embed_tagging_data(fourmomenta_t, scalars_t, ptr_t, cfg.data)
                emb["num_graphs"] = len(ptr_t) - 1  # TaggerWrapper.forward requires this

                # ── Hooks ─────────────────────────────────────────────────────
                # Capture: (a) hidden state h at each layer (for CKA)
                #          (b) local Q, K, V from last block's qkv_linear (for attention)
                captured_hs  = {}   # {layer_idx: (1, n_phys, hidden_ch) tensor}
                captured_qkv = {}   # {"q", "k", "v"}: (1, H, n_phys, head_ch) on device

                def _make_hs_hook(li):
                    def _hook(module, inp, out):
                        # out shape: (1, n_phys, hidden_ch) — leading batch dim from TransformerWrapper
                        captured_hs[li] = out.detach().cpu()
                    return _hook

                def _qkv_hook(module, inp, out):
                    q, k, v = out   # each: (1, H, n_phys, head_ch) — on device
                    captured_qkv["q"] = q.detach()
                    captured_qkv["k"] = k.detach()
                    captured_qkv["v"] = v.detach()

                hooks = []
                hooks.append(model.net.linear_in.register_forward_hook(_make_hs_hook(0)))
                for li, blk in enumerate(model.net.blocks):
                    hooks.append(blk.register_forward_hook(_make_hs_hook(li + 1)))
                hooks.append(
                    model.net.blocks[-1].attention.qkv_linear.register_forward_hook(_qkv_hook)
                )

                _ = model(emb)

                for h in hooks:
                    h.remove()

                if not captured_qkv:
                    n_processed += 1
                    continue

                # ── CKA: mean of physical-token h at each layer ───────────────
                if n_cka < args.n_cka:
                    for li in range(NUM_LAYERS):
                        h = captured_hs.get(li)
                        if h is None:
                            continue
                        # h: (1, n_phys, hidden_ch) → mean over tokens → (hidden_ch,)
                        hs_cka[li].append(
                            h[0].mean(0).numpy().astype(np.float32)
                        )
                    n_cka += 1

                # ── Attention: Q, K from last block's qkv_linear ─────────────
                # For standard transformer, Q/K from qkv_linear are already the
                # actual dot-product keys — no frame transform needed.
                q = captured_qkv["q"]   # (1, H, n_phys, head_ch) on device
                k = captured_qkv["k"]

                scale = q.shape[-1] ** -0.5
                # Per-jet attention: (1, H, k_eff, n_phys)
                q_top_k = q[:, :, :k_eff, :]
                attn = torch.softmax(
                    q_top_k @ k.transpose(-2, -1) * scale, dim=-1
                )  # (1, H, k_eff, n_phys)

                # pT-rank: attention received by each physical particle
                ptrank_vals = attn[:, :, :, :k_eff].squeeze(0).mean(0).mean(0).cpu().numpy()
                # (k_eff,) — mean attention weight received at each pT rank

                label_str = "Top" if labels[bi] else "QCD"
                if label_str == "Top":
                    ptrank_top.append(ptrank_vals)
                else:
                    ptrank_qcd.append(ptrank_vals)

                # IAFormer-style averaged attention maps (K×K, physical-to-physical)
                if n_phys >= K:
                    q_K = q[:, :, :K, :]
                    k_K = k[:, :, :K, :]
                    attn_phys = torch.softmax(
                        q_K @ k_K.transpose(-2, -1) * scale, dim=-1
                    ).squeeze(0).cpu().numpy()  # (H, K, K)
                    attn_accum[label_str]["sum"]   += attn_phys
                    attn_accum[label_str]["count"] += 1
                    attn_accum["All"]["sum"]       += attn_phys
                    attn_accum["All"]["count"]     += 1

                n_processed += 1

            if n_processed % 500 == 0 or n_processed == args.n_jets:
                elapsed = time.time() - t0
                print(f"  [{elapsed:6.0f}s]  {n_processed}/{args.n_jets} jets  "
                      f"cka={n_cka}  top={len(ptrank_top)}  qcd={len(ptrank_qcd)}")

    elapsed = time.time() - t0
    print(f"\nDone. jets={n_processed}  cka={n_cka}  elapsed={elapsed:.0f}s")

    # ── Save and plot CKA ─────────────────────────────────────────────────────
    if n_cka > 0:
        print(f"\nComputing {NUM_LAYERS}×{NUM_LAYERS} CKA matrix from {n_cka} jets...")
        t_cka = time.time()
        cka_matrix = compute_cka_matrix(hs_cka)
        print(f"CKA done in {time.time() - t_cka:.1f}s")
        np.save(output_dir / "cka_lloca_transformer.npy", cka_matrix)
        plot_cka(
            cka_matrix,
            f"LLoCa-Transformer  Linear CKA  (TopTagging, mean h, {n_cka} jets)",
            output_dir / "cka_lloca_transformer.png",
        )

    # ── Save and plot pT-rank attention ───────────────────────────────────────
    if ptrank_top or ptrank_qcd:
        plot_ptrank(
            ptrank_top, ptrank_qcd,
            f"LLoCa-Transformer  pT-rank Attention  "
            f"(TopTagging, last block, {len(ptrank_top)+len(ptrank_qcd)} jets)",
            output_dir / "ptrank_attn_lloca_transformer.png",
        )
        np.savez(
            output_dir / "ptrank_attn_lloca_transformer.npz",
            top=np.array(ptrank_top, dtype=object),
            qcd=np.array(ptrank_qcd, dtype=object),
        )

    # Save and plot averaged attention maps (IAFormer-style per-head heatmaps)
    for cls in ["Top", "QCD", "All"]:
        cnt = attn_accum[cls]["count"]
        if cnt > 0:
            avg_attn = (attn_accum[cls]["sum"] / cnt).astype(np.float32)
            np.save(output_dir / f"avg_attn_lloca_transformer_{cls.lower()}.npy", avg_attn)
            plot_averaged_attention_grid(
                avg_attn, cnt, label=cls, title_prefix="LLoCa-Transformer",
                save_path=output_dir / f"avg_attn_lloca_transformer_{cls.lower()}.png",
            )

    print(f"\nAll results in {output_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="LLoCa-Transformer extended interpretation: CKA + pT-rank attention"
    )
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config",     required=True)
    ap.add_argument("--data-path",  required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-jets",     type=int, default=10000)
    ap.add_argument("--n-cka",      type=int, default=5000)
    args = ap.parse_args()
    main(args)
