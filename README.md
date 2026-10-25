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

> **Note:** This is a cleanup of our original code using Claude. We didn't release the code initially
> since our original code was a bit messy and divided into multiple repos and servers. So, this code
> may contain some errors and is not end-to-end reproducible. We are working on that along with a
> small revision of the paper improving the readability and adding some details. Our goal with this
> release is to provide all the scripts we used for our analysis, since that might be useful for
> someone trying to interpret their own model.
>
> For any queries, please contact us at jay25official@gmail.com.

## Setup

The scripts import model and data code from the repositories the models were trained with. Clone
[lloca-experiments](https://github.com/heidelberg-hepml/lloca-experiments) (or
[tagger-quantization](https://github.com/heidelberg-hepml/tagger-quantization), which provides the
same `experiments` package) and
[lorentz-gatr](https://github.com/heidelberg-hepml/lorentz-gatr) (the `gatr` package used for the
multivector invariants), and put them on the path in this order:

```bash
export PYTHONPATH=path/to/lloca-experiments:path/to/lorentz-gatr
export RUNS=path/to/lloca-experiments/runs     # one folder per model and seed, e.g. $RUNS/topt_lgatr/seed1001
export DATA=data/toptagging_full.npz
python scripts/get_data.py                     # downloads $DATA, 1.5 GB
```

Three scripts run from another folder, as in our runs: `src/part/run_probes_part_topt.py` and
`src/part/run_equivariance_test_part.py` from [particle_transformer](https://github.com/jet-universe/particle_transformer)
(they also need [weaver](https://github.com/hqucms/weaver-core)), and
`src/lgatr/run_equivariance_test.py` from lorentz-gatr.
The scripts load and run the models with Python 3.10, torch 2.11, lgatr 1.4.3, lloca 1.3.5, xformers 0.0.35,
torch-geometric 2.7, hydra-core 1.3, scikit-learn 1.7, clifford 1.5 and matplotlib 3.10. Each
script's docstring gives the command and the arguments the paper used.

## Layout

- `src/lgatr`, `src/lgatr_slim`, `src/transformer` (Vanilla-T and LLoCa-T), `src/part`: extraction
  and analysis scripts per model
- `src/uncertainty`: per-seed bootstraps and the intervals pooled across the three seeds
- `figures`: one script per paper figure
- `scripts/get_data.py`: downloads the top tagging dataset

The figure scripts read `results/` in this layout, which is where the docstring commands write:

```text
results/
  probes/<model>/seed<N>/        probe_data_<model>.npz, probe_results/, bootstrap_torch/
  probes/lgatr/seed<N>/          also probe_data_lgatr_mv*.npz, mv_probe_results/, mv_bootstrap/
  ablations/<variant>/seed<N>/   grade_decomp_full.npz, grade_decomp_scores.npz (slim: grade_decomp_slim*.npz)
  ablations/<variant>_layer/seed<N>/
  interp/<model>/seed<N>/        avg_attn_*.npy, cka_*.npy
  equivariance_gamma5/<model>/seed<N>/
```

`<model>` is `lgatr`, `lgatr_slim`, `lloca_transformer`, `vanilla_transformer` or `part`
(`lloca_transformer_full_event` for the LLoCa-T equivariance run); `<variant>` is `lgatr`,
`lgatr_slim`, `lgatr_bivector_false` or `lgatr_subgroup_false`. The pooled intervals are written
next to the seed folders by `src/uncertainty/compute_combined_bootstrap_ci.py` (probes, ablations,
invariant probes) and `src/uncertainty/combined_equivariance_bootstrap_gamma5.py` (equivariance).

## Paper figures and tables

| Paper | Analysis | Scripts | Figure file |
|---|---|---|---|
| Fig. 1, Tables 8-10 | equivariance and boost sweep | `src/*/run_equivariance_*.py`, `src/uncertainty/combined_equivariance_bootstrap_gamma5.py`, `figures/fig1_equivariance.py` | `main/fig1_equivariance` |
| Fig. 2, Table 3 | final-layer linear probes | `src/*/run_probes_*.py`, `src/*/fit_probes*.py`, `src/uncertainty/bootstrap_probe_npz_torch.py`, `figures/fig2_probe_bar.py` | `main/fig3_probe_bar` |
| Figs. 3, 10 | probe trajectories | as above, `figures/fig3_probe_trajectories.py`, `figures/fig10_probe_trajectories_all.py` | `main/fig2_hs_probe_summary`, `appendix/figA3_hs_probe_extra` |
| Fig. 4, Table 2 | grade ablations for L-GATr | `src/lgatr/run_grade_decomp_lgatr.py`, `figures/fig4_grade_ablation.py` | `main/fig4_grade_decomp` |
| Figs. 5-8 | attention maps | `src/*/run_extended_interp_*.py`, `figures/fig5_8_attention.py` | `appendix/figA1a`-`figA1d` |
| Fig. 9 | layer CKA | as above, `figures/fig9_cka.py` | `appendix/figA2_cka_compare` |
| Figs. 11, 12, Tables 4, 5 | geometric-algebra-invariant probes | `src/lgatr/run_probes_lgatr_mv.py`, `src/lgatr/compute_mv_invariants.py`, `src/lgatr/run_mv_probes_full.py`, `src/uncertainty/bootstrap_invariant_probe_npz_torch.py`, `figures/fig11_invariant_probes_jet.py`, `figures/fig12_invariant_probes_particle.py` | `appendix/figA4a`, `figA4b` |
| Fig. 13 | L-GATr trained without bivectors | `src/lgatr/run_grade_decomp_lgatr.py`, `figures/fig13_no_bivector_ablation.py` | `appendix/figA5a_biv_false` |
| Fig. 14, Table 6 | channel ablations for L-GATr-slim | `src/lgatr_slim/run_grade_decomp_slim.py`, `figures/fig14_slim_ablation.py` | `appendix/figA5b_slim` |
| Fig. 15, Table 7 | L-GATr with five independent grades | `src/lgatr/run_grade_decomp_lgatr.py --interventions all_grades`, `figures/fig15_five_grade_ablation.py` | `appendix/figA5c_subgroup_false` |

The figure scripts write to `figures/main/` and `figures/appendix/` under the file names of the
paper's source. Table 1 is the test AUC reported at the end of training.

## Citation

```bibtex
@article{agarwal2026lorentz,
  title   = {What Do Lorentz-Equivariant Jet Taggers Learn?},
  author  = {Agarwal, Jay and Khare, Siddharth and Kumar, Dhruv},
  journal = {arXiv preprint arXiv:2606.21790},
  year    = {2026}
}
```
