#!/usr/bin/env python
"""
Full mv probe analysis: all 13 targets (7 jet + 6 particle), all grade groups, all layers.

Particle invariants computed on-the-fly per layer from part_mv (50k random sample).
All plots saved as PNG in one output folder.

Usage:
  python run_mv_probes_full.py \
      --inv-data  <path>/probe_data_lgatr_mv_invariants.npz \
      --mv-data   <path>/probe_data_lgatr_mv.npz \
      --scalar-data <path>/probe_data_lgatr.npz \
      --output-dir  <path>/mv_probe_results
"""

import argparse, json, time, warnings, sys
import numpy as np
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, roc_auc_score, accuracy_score
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from compute_mv_invariants import (
    _build_dualization_matrix, _build_grade4_projection, compute_invariants_batch,
)

# ── CLI args ─────────────────────────────────────────────────────────────────
_ap = argparse.ArgumentParser()
_ap.add_argument("--inv-data",    default=None)
_ap.add_argument("--mv-data",     default=None)
_ap.add_argument("--scalar-data", default=None)
_ap.add_argument("--output-dir",  default=None)
_ap.add_argument("--n-train-jet", type=int, default=8000)
_ap.add_argument("--n-test-jet",  type=int, default=2000)
_ap.add_argument("--part-sample", type=int, default=50000)
_args = _ap.parse_args()

# ── config ───────────────────────────────────────────────────────────────────
INV_DATA    = _args.inv_data    or "results/probe_data_lgatr_mv_invariants.npz"
MV_DATA     = _args.mv_data     or "results/probe_data_lgatr_mv.npz"
SCALAR_DATA = _args.scalar_data or "results/probe_data_lgatr.npz"
OUT_DIR     = Path(_args.output_dir or "results/probes_mv_final")
N_TRAIN_JET = _args.n_train_jet
N_TEST_JET  = _args.n_test_jet
PART_SAMPLE = _args.part_sample

N_LAYERS = 13
LAYERS   = np.arange(N_LAYERS)

# Workshop-main groups (contiguous in the 832-feature vector)
GROUPS = ["scalar_like", "vector_like", "bivector_like", "All"]
SLICES = {
    "scalar_like":   (0,   32),   # G0 + G4
    "vector_like":   (32,  560),  # G1 + G3 + Cross
    "bivector_like": (560, 832),  # G2_scalar + G2_pseudo
    "All":           (0,   832),
}
# Appendix: raw grade groups kept for fine-grained diagnostics
GROUPS_APPENDIX = ["G0","G4","G1","G3","Cross","G2_scalar","G2_pseudo"]
SLICES_APPENDIX = {
    "G0":(0,16),"G4":(16,32),"G1":(32,168),"G3":(168,304),
    "Cross":(304,560),"G2_scalar":(560,696),"G2_pseudo":(696,832),
}
SLICES_ALL = {**SLICES, **SLICES_APPENDIX}

COLORS = {
    "scalar_like":"#888","vector_like":"#1f77b4","bivector_like":"#2ca02c","All":"#000",
    "G0":"#bbb","G4":"#999","G1":"#4da6e8","G3":"#ff7f0e","Cross":"#9467bd",
    "G2_scalar":"#5cb85c","G2_pseudo":"#d62728","h_s":"#8c564b",
}
LS  = {
    "scalar_like":":","vector_like":"-","bivector_like":"--","All":"-",
    "G0":":","G4":":","G1":"-","G3":"--","Cross":"-.","G2_scalar":"-","G2_pseudo":"--","h_s":"-",
}
LW  = {
    "scalar_like":2.0,"vector_like":2.2,"bivector_like":2.2,"All":2.4,
    "G0":1.0,"G4":1.0,"G1":1.5,"G3":1.5,"Cross":1.5,"G2_scalar":1.5,"G2_pseudo":1.5,"h_s":2.4,
}

TITLES = {"label":"Top/QCD (AUC)","mass":"Jet mass","mult":"Multiplicity",
          "tau1":"τ₁","tau2":"τ₂","tau3":"τ₃","tau21":"τ₂₁","tau32":"τ₃₂",
          "part_pt":"Particle pT","part_eta":"Particle η (Lorentz check)",
          "part_phi":"Particle φ","part_E":"Particle E",
          "part_quartile":"pT rank quartile","part_dR":"ΔR to jet axis"}
YLABEL = {t: "AUC" if t=="label" else ("Accuracy" if t=="part_quartile" else "R²") for t in TITLES}

# ── probers ──────────────────────────────────────────────────────────────────

def _split(X, y, tr, te):
    sc = StandardScaler()
    return sc.fit_transform(X[tr]), sc.transform(X[te]), y[tr], y[te]

def probe_reg(X, y, tr, te):
    Xtr, Xte, ytr, yte = _split(X, y, tr, te)
    return float(r2_score(yte, Ridge(alpha=10.).fit(Xtr, ytr).predict(Xte)))

def probe_bin(X, y, tr, te):
    Xtr, Xte, ytr, yte = _split(X, y, tr, te)
    clf = LogisticRegression(max_iter=1000, C=0.1).fit(Xtr, ytr.astype(int))
    return float(roc_auc_score(yte, clf.predict_proba(Xte)[:,1]))

def probe_multi(X, y, tr, te):
    Xtr, Xte, ytr, yte = _split(X, y, tr, te)
    clf = LogisticRegression(max_iter=1000, C=0.1).fit(Xtr, ytr.astype(int))
    return float(accuracy_score(yte.astype(int), clf.predict(Xte)))

def fit_groups(inv, y, kind, tr, te):
    fn = {"binary": probe_bin, "multi": probe_multi, "regression": probe_reg}[kind]
    return {g: fn(inv[:, s:e], y, tr, te) for g, (s, e) in SLICES_ALL.items()}

def fit_hs(hs, y, kind, tr, te):
    fn = {"binary": probe_bin, "multi": probe_multi, "regression": probe_reg}[kind]
    return fn(hs, y, tr, te)

# ── plot helpers ─────────────────────────────────────────────────────────────

def leg(fig, grps, **kw):
    handles = [plt.Line2D([0],[0], color=COLORS[g], linestyle=LS[g], linewidth=LW[g]) for g in grps]
    fig.legend(handles, grps, loc="lower right", ncol=5, fontsize=9,
               bbox_to_anchor=(0.99, 0.01), **kw)

def traj_panel(ax, scores, hs_vals, tgt, grp_list=None):
    if grp_list is None: grp_list = GROUPS
    for g in grp_list:
        ax.plot(LAYERS, [scores[li][g] for li in range(N_LAYERS)],
                color=COLORS[g], linestyle=LS[g], linewidth=LW[g])
    ax.plot(LAYERS, hs_vals, color=COLORS["h_s"], linestyle=LS["h_s"], linewidth=2.2, zorder=10)
    ax.set_xlabel("Layer", fontsize=9); ax.set_ylabel(YLABEL[tgt], fontsize=9)
    ax.set_title(TITLES[tgt], fontsize=10); ax.set_xticks(LAYERS); ax.grid(True, alpha=0.3)

def heatmap_panel(ax, scores, tgt):
    grps = [g for g in GROUPS if g != "All"]
    data = np.array([[scores[li][g] for li in range(N_LAYERS)] for g in grps])
    vmin = max(0, data.min())
    im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=vmin, vmax=data.max())
    ax.set_xticks(LAYERS); ax.set_xticklabels(LAYERS, fontsize=6)
    ax.set_yticks(np.arange(len(grps))); ax.set_yticklabels(grps, fontsize=8)
    ax.set_xlabel("Layer", fontsize=9); ax.set_title(TITLES[tgt], fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.04)
    for gi, g in enumerate(grps):
        for li in range(N_LAYERS):
            v = scores[li][g]
            ax.text(li, gi, f"{v:.2f}", ha="center", va="center", fontsize=4.5,
                    color="white" if v < (vmin+data.max())/2 else "black")

def save(fig, name):
    fig.savefig(OUT_DIR / f"{name}.png", dpi=150)
    plt.close(fig)
    print(f"  {name}.png")

# ── main ─────────────────────────────────────────────────────────────────────

OUT_DIR.mkdir(exist_ok=True)
t0 = time.time()

print("Building invariant matrices...")
dual_mat = _build_dualization_matrix()
gp4_mat  = _build_grade4_projection()

print("Loading data...")
inv_d  = np.load(INV_DATA)
mv_d   = np.load(MV_DATA)
sc_d   = np.load(SCALAR_DATA)

jet_inv      = inv_d["jet_inv"]           # (13, 10000, 832)
jet_reps_hs  = sc_d["jet_reps"]           # (13, 10000, 32)
part_mv_all  = mv_d["particle_mv_reps"]   # (13, ~426k, 16, 16) float16
part_reps_hs = sc_d["particle_reps"]      # (13, ~426k, 32)

n_jets  = jet_inv.shape[1]
n_parts = part_mv_all.shape[1]
print(f"  {n_jets} jets, {n_parts} particles")

# jet train/test split
jet_tr = np.arange(N_TRAIN_JET)
jet_te = np.arange(N_TRAIN_JET, N_TRAIN_JET + N_TEST_JET)
n_jet  = N_TRAIN_JET + N_TEST_JET

# particle sample
rng = np.random.default_rng(42)
pidx = rng.choice(n_parts, size=PART_SAMPLE, replace=False); pidx.sort()
p_tr = np.arange(int(0.8 * PART_SAMPLE))
p_te = np.arange(int(0.8 * PART_SAMPLE), PART_SAMPLE)
print(f"  particle sample: {PART_SAMPLE} ({len(p_tr)} train / {len(p_te)} test)")

# targets
_tau32 = (sc_d["jet_tau32"] if "jet_tau32" in sc_d.files
          else sc_d["jet_tau3"] / (sc_d["jet_tau2"] + 1e-10))
jet_tgts = {
    "label" : ("binary",     sc_d["jet_labels"][:n_jet].astype(float)),
    "mass"  : ("regression", sc_d["jet_mass"][:n_jet]),
    "mult"  : ("regression", sc_d["jet_multiplicity"][:n_jet].astype(float)),
    "tau1"  : ("regression", sc_d["jet_tau1"][:n_jet]),
    "tau2"  : ("regression", sc_d["jet_tau2"][:n_jet]),
    "tau3"  : ("regression", sc_d["jet_tau3"][:n_jet]),
    "tau21" : ("regression", sc_d["jet_tau21"][:n_jet]),
    "tau32" : ("regression", _tau32[:n_jet]),
}
part_tgts = {
    "part_pt"       : ("regression", sc_d["particle_pt"][pidx]),
    "part_eta"      : ("regression", sc_d["particle_eta"][pidx]),
    "part_phi"      : ("regression", sc_d["particle_phi"][pidx]),
    "part_E"        : ("regression", sc_d["particle_E"][pidx]),
    "part_quartile" : ("multi",      sc_d["particle_pt_rank_quartile"][pidx].astype(float)),
    "part_dR"       : ("regression", sc_d["particle_dR_to_axis"][pidx]),
}

# result containers
jet_sc  = {t: [] for t in jet_tgts}
jet_hs  = {t: [] for t in jet_tgts}
part_sc = {t: [] for t in part_tgts}
part_hs = {t: [] for t in part_tgts}

# ── fit probes ────────────────────────────────────────────────────────────────
print("\nFitting probes...")
for li in range(N_LAYERS):
    # jet
    jinv = jet_inv[li, :n_jet]
    jhs  = jet_reps_hs[li, :n_jet]
    for t, (kind, y) in jet_tgts.items():
        jet_sc[t].append(fit_groups(jinv, y, kind, jet_tr, jet_te))
        jet_hs[t].append(fit_hs(jhs, y, kind, jet_tr, jet_te))

    # particle — compute invariants on the fly for sampled particles
    pmv  = part_mv_all[li, pidx].astype(np.float32)   # (50k, 16, 16)
    pinv = compute_invariants_batch(pmv, dual_mat, gp4_mat)  # (50k, 832)
    phs  = part_reps_hs[li, pidx]
    for t, (kind, y) in part_tgts.items():
        part_sc[t].append(fit_groups(pinv, y, kind, p_tr, p_te))
        part_hs[t].append(fit_hs(phs, y, kind, p_tr, p_te))

    print(f"  L{li:2d} | jet tau2 All={jet_sc['tau2'][-1]['All']:.3f} "
          f"| part_eta All={part_sc['part_eta'][-1]['All']:.4f} "
          f"[{time.time()-t0:.0f}s]")

# save JSON
all_res = {}
for li in range(N_LAYERS):
    all_res[f"L{li}"] = {}
    for t in jet_tgts:
        all_res[f"L{li}"][t] = {**jet_sc[t][li], "h_s": jet_hs[t][li]}
    for t in part_tgts:
        all_res[f"L{li}"][t] = {**part_sc[t][li], "h_s": part_hs[t][li]}
with open(OUT_DIR / "results.json", "w") as f:
    json.dump(all_res, f, indent=2)

# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerating plots...")

jt = list(jet_tgts.keys())   # 8 jet targets (label, mass, mult, tau1-3, tau21, tau32)
pt = list(part_tgts.keys())  # 6 particle targets

# ── 01 jet trajectories ───────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(22, 9))
fig.suptitle("LGATr Lorentz-invariant probes — jet level (workshop grade groups per layer)", fontsize=13)
for ax, t in zip(axes.flat, jt): traj_panel(ax, jet_sc[t], jet_hs[t], t)
leg(fig, GROUPS+["h_s"])
fig.tight_layout(); save(fig, "01_jet_trajectories")

# ── 02 particle trajectories ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 9))
fig.suptitle("LGATr Lorentz-invariant probes — particle level (all grade groups per layer)", fontsize=13)
for ax, t in zip(axes.flat, pt): traj_panel(ax, part_sc[t], part_hs[t], t)
leg(fig, GROUPS+["h_s"])
fig.tight_layout(); save(fig, "02_particle_trajectories")

# ── 03 jet heatmaps ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(24, 8))
fig.suptitle("LGATr — grade × layer heatmaps (jet)", fontsize=13)
for ax, t in zip(axes.flat, jt): heatmap_panel(ax, jet_sc[t], t)
fig.tight_layout(); save(fig, "03_jet_heatmaps")

# ── 04 particle heatmaps ─────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(20, 8))
fig.suptitle("LGATr — grade × layer heatmaps (particle)", fontsize=13)
for ax, t in zip(axes.flat, pt): heatmap_panel(ax, part_sc[t], t)
fig.tight_layout(); save(fig, "04_particle_heatmaps")

# ── 05 delta over h_s (jet) ──────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(22, 8))
fig.suptitle("LGATr — Δ score = grade group − h_s baseline (jet)", fontsize=13)
for ax, t in zip(axes.flat, jt):
    for g in GROUPS:
        ax.plot(LAYERS, [jet_sc[t][li][g]-jet_hs[t][li] for li in range(N_LAYERS)],
                color=COLORS[g], linestyle=LS[g], linewidth=LW[g])
    ax.axhline(0, color="black", lw=1.2)
    ax.set_xlabel("Layer",fontsize=9); ax.set_ylabel("Δ score",fontsize=9)
    ax.set_title(TITLES[t],fontsize=10); ax.set_xticks(LAYERS); ax.grid(True,alpha=0.3)
leg(fig, GROUPS)
fig.tight_layout(); save(fig, "05_jet_delta_over_hs")

# ── 06 G1 vs G3 ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(22, 8))
fig.suptitle("LGATr — G1 vs G3: Hodge duality test", fontsize=13)
for ax, t in zip(axes.flat, jt):
    g1 = np.array([jet_sc[t][li]["G1"] for li in range(N_LAYERS)])
    g3 = np.array([jet_sc[t][li]["G3"] for li in range(N_LAYERS)])
    ax.plot(LAYERS, g1, color=COLORS["G1"], linewidth=2, label="G1")
    ax.plot(LAYERS, g3, color=COLORS["G3"], linewidth=2, linestyle="--", label="G3")
    ax.fill_between(LAYERS, g1, g3, alpha=0.2, color="purple")
    ax2 = ax.twinx()
    ax2.plot(LAYERS, g1-g3, color="purple", lw=1, ls=":", alpha=0.7)
    ax2.set_ylabel("G1−G3", fontsize=7, color="purple")
    ax2.tick_params(axis="y", labelcolor="purple", labelsize=7)
    ax.set_xlabel("Layer",fontsize=9); ax.set_ylabel(YLABEL[t],fontsize=9)
    ax.set_title(TITLES[t],fontsize=10); ax.set_xticks(LAYERS); ax.grid(True,alpha=0.3)
fig.legend([plt.Line2D([0],[0],color=COLORS["G1"],lw=2),
            plt.Line2D([0],[0],color=COLORS["G3"],lw=2,ls="--")],
           ["G1","G3"], loc="lower right", fontsize=11)
fig.tight_layout(); save(fig, "06_g1_vs_g3_appendix")

# ── 07 grade-2 spotlight ─────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(22, 8))
fig.suptitle("LGATr — Grade-2 spotlight (scalar vs pseudoscalar vs G1 vs All)", fontsize=13)
spot = ["G2_scalar","G2_pseudo","G1","All"]
for ax, t in zip(axes.flat, jt):
    traj_panel(ax, jet_sc[t], jet_hs[t], t, grp_list=spot)
leg(fig, spot+["h_s"])
fig.tight_layout(); save(fig, "07_grade2_spotlight_appendix")

# ── 08 info emergence (normalised) ───────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(22, 8))
fig.suptitle("LGATr — Information emergence: score / max_layer_score (jet)", fontsize=13)
eg = ["scalar_like","vector_like","bivector_like","All","h_s"]
for ax, t in zip(axes.flat, jt):
    for g in eg:
        r = np.array(jet_hs[t]) if g=="h_s" else np.array([jet_sc[t][li][g] for li in range(N_LAYERS)])
        rp = np.clip(r, 0, None); mx = rp.max()
        ax.plot(LAYERS, rp/mx if mx>0 else rp, color=COLORS[g], linestyle=LS[g], linewidth=LW[g])
    ax.set_xlabel("Layer",fontsize=9); ax.set_ylabel("Score / max",fontsize=9)
    ax.set_title(TITLES[t],fontsize=10); ax.set_xticks(LAYERS); ax.set_ylim(-0.05,1.15); ax.grid(True,alpha=0.3)
axes.flat[-1].set_visible(False); leg(fig, eg)
fig.tight_layout(); save(fig, "08_info_emergence")

# ── 09 peak layers heatmap ───────────────────────────────────────────────────
all_tgts  = jt + pt
all_sc    = {**{t: jet_sc[t] for t in jt}, **{t: part_sc[t] for t in pt}}
fig, ax = plt.subplots(figsize=(18, 4))
mat = np.array([[np.argmax([all_sc[t][li][g] for li in range(N_LAYERS)]) for t in all_tgts]
                for g in GROUPS])
im = ax.imshow(mat, cmap="plasma", vmin=0, vmax=12, aspect="auto")
ax.set_xticks(np.arange(len(all_tgts)))
ax.set_xticklabels([TITLES[t].split("(")[0].strip() for t in all_tgts], rotation=35, ha="right", fontsize=9)
ax.set_yticks(np.arange(len(GROUPS))); ax.set_yticklabels(GROUPS, fontsize=10)
plt.colorbar(im, ax=ax, label="Layer of peak score")
for gi,g in enumerate(GROUPS):
    for ti,t in enumerate(all_tgts):
        ax.text(ti, gi, str(mat[gi,ti]), ha="center", va="center", fontsize=8,
                color="white" if mat[gi,ti]<7 else "black")
ax.set_title("Layer of peak score — all grade groups × all 14 targets", fontsize=12)
fig.tight_layout(); save(fig, "09_peak_layers")

# ── 10 final layer bars (all 13 targets) ─────────────────────────────────────
all_hs = {**{t: jet_hs[t] for t in jt}, **{t: part_hs[t] for t in pt}}
fig, ax = plt.subplots(figsize=(22, 5))
grps_b = GROUPS+["h_s"]; x = np.arange(len(all_tgts)); w = 0.075
for gi,g in enumerate(grps_b):
    vals = [all_sc[t][-1][g] if g!="h_s" else all_hs[t][-1] for t in all_tgts]
    ax.bar(x+(gi-len(grps_b)/2)*w+w/2, vals, w, label=g, color=COLORS[g], alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([TITLES[t].split("(")[0].strip() for t in all_tgts], rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Score"); ax.set_title("Final layer (L=12): all grade groups × all 14 targets")
ax.legend(fontsize=8, ncol=5); ax.grid(axis="y", alpha=0.3); ax.set_ylim(0, 1.1)
fig.tight_layout(); save(fig, "10_final_layer_bars")

# ── 11 summary grid (all 13) ─────────────────────────────────────────────────
fig, axes = plt.subplots(3, 5, figsize=(26, 15))
fig.suptitle("LGATr Lorentz-invariant probes — all 14 targets", fontsize=15)
for ax, t in zip(axes.flat, all_tgts):
    sc_t  = all_sc[t]
    hs_t  = all_hs[t]
    traj_panel(ax, sc_t, hs_t, t)
    ax.tick_params(labelsize=7)
for ax in axes.flat[len(all_tgts):]: ax.set_visible(False)
leg(fig, GROUPS+["h_s"])
fig.tight_layout(); save(fig, "11_summary_grid")

# ── 12 part_eta spotlight — Lorentz invariance sanity check ──────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_title("Lorentz invariance sanity check: particle η probe (should be ≈ 0)", fontsize=12)
for g in GROUPS+["h_s"]:
    if g == "h_s": vals = part_hs["part_eta"]
    else: vals = [part_sc["part_eta"][li][g] for li in range(N_LAYERS)]
    ax.plot(LAYERS, vals, color=COLORS[g], linestyle=LS[g], linewidth=LW[g], label=g)
ax.axhline(0, color="black", lw=1.2, ls="--")
ax.set_xlabel("Layer"); ax.set_ylabel("R²")
ax.set_xticks(LAYERS); ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, ncol=2)
fig.tight_layout(); save(fig, "12_lorentz_sanity_part_eta")

# ── print final table ─────────────────────────────────────────────────────────
print(f"\n{'='*85}")
print(f"{'Target':>16}  {'h_s':>6}  {'scalar':>6}  {'vector':>6}  {'bivec':>6}  {'All':>6}")
print("-"*85)
for t in all_tgts:
    s = all_sc[t][-1]; h = all_hs[t][-1]
    print(f"{t:>16}  {h:>6.3f}  {s['scalar_like']:>6.3f}  {s['vector_like']:>6.3f}  "
          f"{s['bivector_like']:>6.3f}  {s['All']:>6.3f}")

print(f"\nDone in {time.time()-t0:.0f}s. All plots in {OUT_DIR}/")
