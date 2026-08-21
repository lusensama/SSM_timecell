# Response figures — the selected set

The nine plots that are actually going into the response, and nothing else:
the figure files, the numbers behind them, and the code that turns one into the
other. Self-contained — no `training/` tree, no `.pt` checkpoints, no GPU.

```bash
module load python3.11-anaconda/2024.02
source /sw/pkgs/arc/python3.11-anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate ebt
python scripts/make_all.py            # ~9 s, rebuilds all of it
```

Verified: `data/` + `scripts/` were copied to an empty directory with the
project absent and `make_all.py` reproduced every PNG.

## The figures

| File | Answers | Shows | Data |
| --- | --- | --- | --- |
| `fig5d_lap_confusion` | R3.3 | Lap identity decoded at an EXACT absolute timestep, so every lap class shares the same elapsed time and the published panel's time confound is gone by construction. 98.6% balanced, chance 39.5%, 108,030 states. | `fig5de_confusion.csv` |
| `fig5e_lap_discriminant` | R3.3 | The same states on the decoder's own discriminant axes. Basis and boundary fit on a train half; the plotted points are the held-out half — 98.2% in 2-D against 25% chance. | `fig5de_projection.csv` |
| `fig_R22_R24_retiming` | R2.2, R2.4 | Accuracy vs D_test, 2 × 2: **a** D_train = 30 zero-shot, **b** D_train = 30 readout-only, **c** D_train = 60 zero-shot, **d** D_train = 60 readout-only. 1,920 runs. | `exp3_retime_cells.csv`, `exp3_retime_basemodel.csv` |
| `fig_R22_R24_retiming_annotated` | R2.2, R2.4 | The same figure with its text layer — title, subtitle, panel titles, provenance stamp. | same |
| `fig_R23_lstm_retiming` | R2.3 | The retraining curves: the LSTM stays flat under freeze-and-retime while both SSM arms climb. | `exp6_retime_curves.csv` |
| `fig_baseline_hippo` | — (baseline) | The reference model: the HiPPO-LegS SSM's 50 units' delay-period activity, at the trained delay of 30 and retimed to 100. Plain heatmaps, sorted units, nothing drawn on top. | `exp1_heatmap_matrices.npz` |
| `fig_R31_cascade_heatmaps` | R2.1, R3.1 | Delay-period activity at init and after training, all 9 initializations. | `exp1_heatmap_matrices.npz` |
| `fig_R32_fig3_relative_change` | R3.2, **manuscript Fig. 3b/3c** | The submitted panel re-measured over 10 seeds: elementwise \|Δθ\|/\|θ_init\| for Λ, B, C, HiPPO vs random. | `fig3_relative_change.npz`, `fig3_relative_change_stats.csv` |
| `fig_R33_lap_heatmap` | R3.3 | The sequential basis renormalises to each lap though durations vary six-fold. | `exp5_heatmap_matrices.npz` |

Each is a PNG at 200 dpi (300 for 5d/5e, their original setting) plus a vector PDF.

`fig_baseline_hippo` is new. It is not a recomputation: the two matrices are the
`hippo__trained_d30` and `hippo__retimed_d100` arrays already in `data/` — the
first is the exact cell at the bottom left of `fig_R31_cascade_heatmaps` — drawn
at full width instead of the 1.5 in a 9-wide grid allows. Same sort order, same
colormap, same normalization; only the canvas changes. Each panel is sorted
independently, as everywhere else in this folder, so the two panels share no row
correspondence.

`fig_R23_lstm_retiming` is the only figure here that differs from the one it was
selected from. In `figures/response/` it is a three-panel figure — retraining
curves | gain distributions | freeze verification — and only the **first panel**
is wanted here, so the other two are dropped. The curve-drawing code is the
original's byte-for-byte; the only change is the standalone size and an x limit
tightened to 2,520, since the axis no longer has to leave room for the panels
that were to its right. The four CSVs behind the dropped panels
(`exp6_retime_units.csv`, `exp6_retime_stats.csv`, `exp6_retime_welch.csv`,
`exp6_freeze_check.csv`) are not carried; they stay in
`figures/response/data/`. Every other PNG in this folder is byte-identical to
the one it was selected from — `figures/response/` for the five other `fig_R*`,
`figures/exp5_random/fig5de_n5000_trained/` for 5d and 5e.

### Text layer

The `fig_R*` figures ship **bare**: no figure title, no subtitle, no panel
titles, no provenance stamp, no explanatory prose — so a text layer can be added
downstream. What is left is the plot: axes, tick labels, axis labels, legends,
direct series labels, reference-line labels (`chance 33.3%`) and numeric value
labels. `fig_R22_R24_retiming_annotated` is the one exception, kept because it
is the version already in use; `make_all.py` builds it by rerunning the same
code with `RESP_BARE=0`. To get annotated builds of the others:

```bash
RESP_BARE=0 python scripts/plot_exp1.py     # and plot_baseline_hippo / plot_exp5 / plot_exp6 / plot_fig3
```

Fig. 5d and 5e are **not** bare — they are drawn by their original run's own
code with stock matplotlib and keep their original titles.

## `data/` — the raw numbers

| File | Rows | Feeds |
| --- | --- | --- |
| `fig5de_confusion.csv` | 16 | 5d — the pooled 4 × 4 confusion, counts and row percentages |
| `fig5de_projection.csv` | 9,002 | 5e — every plotted point: the 80-D state at t = 90 (seed 2) projected to 2-D, both the train half that fit the basis and the held-out half that is drawn |
| `fig5de_meta.json` | — | 5d/5e accuracies, chance levels, axis names, sample counts |
| `exp3_retime_cells.csv` | 64 | R2.2/R2.4 — one row per (D_train, phase, init, D_test) cell |
| `exp3_retime_basemodel.csv` | 1,280 | the per-base-model points behind those cells |
| `exp6_retime_curves.csv` | 297 | R2.3 — the retraining curves, 9 units × 3 architectures |
| `exp1_heatmap_matrices.npz` | 108 arrays | the baseline panel and the R2.1/R3.1 grid: 9 inits × 2 phases × (matrix, time-sorted matrix, peak counts, seed) |
| `exp1_heatmap_matrices.csv` | 144,000 | the same matrices, long-form |
| `exp5_heatmap_matrices.npz` | 12 arrays | R3.3 lap-phase heatmaps |
| `exp5_heatmap_matrices.csv` | 51,200 | the same, long-form |
| `exp5_lap_selectivity.csv` | 160 | per-unit lap selectivity behind that figure |
| `fig3_relative_change.npz` | 8 arrays | Fig. 3b/3c — every elementwise ratio, Λ/Λ̄/B/C × HiPPO/random |
| `fig3_relative_change.csv` | 55,000 | the same, long-form |
| `fig3_relative_change_stats.csv` | 88 | per-parameter means, medians, percentiles, Welch tests |

The `.csv` twins of the three `.npz` bundles are not read by any plot script —
they are there so the numbers can be inspected without numpy.

## `scripts/`

```
scripts/
  make_all.py            rebuild everything from data/  <- the entry point
  style.py               shared marks, palette, type, RESP_BARE
  common.py              paths, CSV reader, mode ordering and labels
  plot_fig5de.py         5d, 5e
  plot_baseline_hippo.py fig_baseline_hippo
  plot_exp1.py           fig_R31_cascade_heatmaps
  plot_exp3.py           fig_R22_R24_retiming (+ annotated)
  plot_exp5.py           fig_R33_lap_heatmap
  plot_exp6.py           fig_R23_lstm_retiming
  plot_fig3.py           fig_R32_fig3_relative_change
  provenance/            the extractors -- NEED the run tree, not to replot
```

`plot_exp1.py`, `plot_exp5.py` and `plot_exp6.py` are trimmed copies of
`figures/response/scripts/`: the figure functions this folder does not carry
were deleted, the ones it does are byte-for-byte the originals. `plot_exp3.py`
and `plot_fig3.py` are unmodified. `plot_exp6.py` keeps only the first panel of
its three-panel original, as described above.

### `scripts/provenance/` — how `data/` was made

These read the original run artifacts and rewrite `data/`. They need
`training/` and `figures/` and are **not** needed to rebuild any figure.

| Script | Reads | Writes |
| --- | --- | --- |
| `extract_exp1.py` | `training/exp1/`, per-unit state caches | `exp1_*` |
| `extract_exp3.py` | `figures/exp3/exp3_retime_grid_units.csv` | `exp3_*` |
| `extract_exp5.py` | `figures/exp5_random/`, `training/exp5_*` | `exp5_*` |
| `extract_exp6.py` | `training/exp6_lstm_*`, slurm logs, checkpoint state_dicts | `exp6_*` |
| `extract_fig3.py` | `training/exp1/{hippo,rand_complex}_s<seed>/.../{initial,final}_*.pt` parameter tensors | `fig3_*` |
| `export_exp5_heatmap.py` | `training/exp5_random/seed_*/best_eval.pt` — **the only one that has to roll a model out** | `exp5_heatmap_matrices.*` |
| `extract_fig5de.py` | the 6.7 GB rollout cache under `figures/exp5_random/fig5de_n5000/_cache/` | `fig5de_*` |
| `plot_fig5de_lap_svm.py` | the exp5_random checkpoints — the original 5d/5e run, kept for reference | that cache + the run's own figures |

`extract_fig5de.py` is new and written for this folder. Fig. 5d/5e were the one
pair with no CSV behind them: their upstream artifact is ~1 GB of 80-D state per
(kind, seed). It reduces that to the two small tables above and **asserts** the
balanced accuracies it recomputes against the ones the original run wrote to
`fig5de_summary.json` — 98.17% for the LDA panel, 25.00% for PCA — so the
reduction is checked, not assumed. It also replays the qualifying-timestep test
on `lap_t` alone to recover panel d's "11 exact timesteps" (11 distinct
timesteps per seed; 10 + 10 + 11 = the 31 seed-timestep decodes that
`summary.json` records as `n_timesteps`).

## Two things to know

1. **The freeze verification is no longer shown.** It was panel c of the
   three-panel original, and its numbers are not the letter's 0.120 / 0.131:
   both agree the frozen blocks moved by exactly 0.0, but for the trainable side
   `figures/response/` computes \|Δθ\|_F/\|θ\|_F per parameter tensor, giving
   LSTM `readout.weight` 0.237 against SSM `C_tilde` 0.228 — same order, the
   letter's qualitative claim, a different normalization. The letter's two
   numbers have no saved artifact behind them. If the letter keeps that claim,
   the 162 per-parameter values are in `figures/response/data/exp6_freeze_check.csv`.

2. **Two statistics are both called "relative parameter change".** The submitted
   Fig. 3b histograms the ELEMENTWISE ratio \|Δθ\|/\|θ_init\|, which is what
   `fig_R32_fig3_relative_change` plots. The `plasticity` field in
   `exp1_summary.jsonl` is a ratio of WHOLE-MATRIX norms and gives different
   ratios (Λ 1.9× vs 12.4×, B 1.7× vs 1.1×). On the quantity Fig. 3b actually
   plots, the manuscript's ordering holds for both Λ and B. Quote the
   elementwise numbers and do not mix the two.

## Colours

Taken from the repo, not invented here. HiPPO is **tab:red `#d62728`** and
random **tab:blue `#1f77b4`** wherever colour encodes the initialization
(`fig_R22_R24_retiming`, the SSM arms of `fig_R23_*`, `fig_R32_fig3_relative_change`);
the LSTM arm takes aqua. All activity heatmaps use **`jet`** with
`interpolation="none"`, matching seven of the repo's eight activity `imshow`
calls. Unit ordering in every heatmap is the manuscript's `sort_freq_resp`
(`utils/utils_analysis.py:13`) — grouped by peak count, then peak time — with
sort orders taken from a held-out half of the episodes so no diagonal can be an
artifact of sorting the data being displayed. `scripts/style.py` records the
file and line each colour comes from. The fuller account is in
`figures/response/README.md`.

## Where this came from

`figures/response/` holds all 17 response-letter figures and all 39 CSVs. This
folder is the subset going into the response, plus the 5d/5e pair from
`figures/exp5_random/fig5de_n5000_trained/`, which is not part of that set.
Nothing here supersedes anything there.
