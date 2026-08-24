# Figures

The nine plots that are found in the main text or SI:
the figure files, the numbers behind them, and the code that turns one into the
other. Self-contained — no `training/` tree, no `.pt` checkpoints, no GPU.

```bash
module load python3.11-anaconda/2024.02
source /sw/pkgs/arc/python3.11-anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate ebt
python scripts/make_all.py            # ~9 s, rebuilds all of it
```

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
```

`plot_exp1.py`, `plot_exp5.py` and `plot_exp6.py` are trimmed copies of
`figures/response/scripts/`: the figure functions this folder does not carry
were deleted, the ones it does are byte-for-byte the originals. `plot_exp3.py`
and `plot_fig3.py` are unmodified. `plot_exp6.py` keeps only the first panel of
its three-panel original, as described above.

