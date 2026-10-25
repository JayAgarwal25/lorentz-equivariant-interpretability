<div align="center">

## What Do Lorentz-Equivariant Jet Taggers Learn?

[![arXiv](https://img.shields.io/badge/paper-arxiv.2606.21790-B31B1B.svg)](https://arxiv.org/abs/2606.21790)

</div>

This repository contains the analysis code for the paper [*What Do Lorentz-Equivariant Jet Taggers
Learn?*](https://arxiv.org/abs/2606.21790) by Jay Agarwal, Siddharth Khare and Dhruv Kumar
(AI4Physics Workshop, ICML 2026). We study five top tagging models (L-GATr, L-GATr-slim, LLoCa-T,
Vanilla-T and ParT) with equivariance tests, linear probes on the scalar channels,
geometric-algebra-invariant probes on the multivector channels, grade ablations, attention maps and
CKA, three training seeds each.

## Layout

- `src/lgatr`, `src/lgatr_slim`, `src/transformer` (Vanilla-T and LLoCa-T), `src/part`: extraction
  and analysis scripts per model
- `src/uncertainty`: bootstrap intervals pooled across the three seeds
- `figures`: one script per paper figure, reading `results/<analysis>/<model>/seed<N>/`
- `scripts/get_data.py`: downloads the top tagging dataset (`data/toptagging_full.npz`, 1.5 GB)

All five models are trained with
[lloca-experiments](https://github.com/heidelberg-hepml/lloca-experiments). The scripts import model
code from it and from [lorentz-gatr](https://github.com/heidelberg-hepml/lorentz-gatr) (L-GATr),
[tagger-quantization](https://github.com/heidelberg-hepml/tagger-quantization) (L-GATr-slim) and
[particle_transformer](https://github.com/jet-universe/particle_transformer) (ParT); each script's
docstring names the repository it runs from.

## Paper figures and tables

| Paper | Analysis | Scripts |
|---|---|---|
| Fig. 1, Tables 8-10 | equivariance and boost sweep | `src/*/run_equivariance_*.py`, `src/uncertainty/bootstrap_equivariance_npz.py`, `src/uncertainty/combined_equivariance_bootstrap_gamma5.py`, `figures/fig1_equivariance.py` |
| Figs. 2, 3, 10, Table 3 | linear probes on the scalar channels | `src/*/run_probes_*.py`, `src/*/fit_probes*.py`, `src/uncertainty/bootstrap_probe_npz_torch.py`, `figures/fig2_probe_bar.py`, `figures/fig3_probe_trajectories.py`, `figures/fig10_probe_trajectories_all.py` |
| Fig. 4, Table 2 | grade ablations for L-GATr | `src/lgatr/run_grade_decomp_lgatr.py`, `src/lgatr/bootstrap_grade_significance.py`, `figures/fig4_grade_ablation.py` |
| Figs. 5-9 | attention maps and CKA | `src/*/run_extended_interp_*.py`, `figures/fig5_8_attention.py`, `figures/fig9_cka.py` |
| Figs. 11, 12, Tables 4, 5 | geometric-algebra-invariant probes | `src/lgatr/run_probes_lgatr_mv.py`, `src/lgatr/compute_mv_invariants.py`, `src/lgatr/run_mv_probes_full.py`, `src/uncertainty/bootstrap_invariant_probe_npz_torch.py`, `figures/fig11_invariant_probes_jet.py`, `figures/fig12_invariant_probes_particle.py` |
| Fig. 13 | L-GATr trained without bivectors | `src/lgatr/run_grade_decomp_lgatr.py`, `figures/fig13_no_bivector_ablation.py` |
| Fig. 14, Table 6 | channel ablations for L-GATr-slim | `src/lgatr_slim/run_grade_decomp_slim.py`, `figures/fig14_slim_ablation.py` |
| Fig. 15, Table 7 | L-GATr with five independent grades | `src/lgatr/run_grade_decomp_lgatr.py --interventions all_grades`, `figures/fig15_five_grade_ablation.py` |

`src/uncertainty/compute_combined_bootstrap_ci.py` pools the per-seed bootstrap draws into the
intervals shown in the figures and tables. Table 1 is the test AUC reported at the end of training.

## Citation

```bibtex
@article{agarwal2026lorentz,
  title   = {What Do Lorentz-Equivariant Jet Taggers Learn?},
  author  = {Agarwal, Jay and Khare, Siddharth and Kumar, Dhruv},
  journal = {arXiv preprint arXiv:2606.21790},
  year    = {2026}
}
```
