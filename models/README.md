# Released checkpoints

Three models, each the best of its family under a clean re-evaluation rather
than the accuracy recorded during training. All are `state_dict` files; load
them with the matching entry point below.

| file | task | script | selected by |
| --- | --- | --- | --- |
| `3stim_hippo_best.pt` | 3-stim interval discrimination, delay 30 | `train_and_plot_3stim.py` | 89.45% |
| `3stim_rand_complex_best.pt` | same, random-complex init | `train_and_plot_3stim.py` | 88.45% |
| `lap_counting_best.pt` | lap counting under randomised timing | `train_landmark_laps.py` (`--env random`) | 94.53% mean over K=2..16 |

## Why these and not the recorded best

The per-run `best_eval_acc_<seed>.txt` files are a **maximum over 25 periodic
evaluations of 1000 episodes each**, every run drawn from its own seed. At ~88%
accuracy a single 1000-episode eval has a standard error near 1 point, so that
maximum is biased upward by roughly 1.5-2 points and the runs are not mutually
comparable. Re-evaluating every candidate on one common seed changed the
ranking: `hippo_s6` recorded the highest value (90.3%) and is *not* the best
model -- on identical episodes it places second.

### 3-stim: all candidates, common seed 1234, 2000 episodes, delay 30

| candidate | clean acc | recorded | retimed d100 |
| --- | --- | --- | --- |
| **hippo_s2 (shipped)** | **89.45%** | 89.0 | 82.3 |
| hippo_s6 | 89.30% | 90.3 | 78.3 |
| **rand_complex_s2 (shipped)** | **88.45%** | 89.6 | 80.4 |
| rand_complex published (`extra_models/`) | 88.10% | - | 77.9 |
| hippo published (`3stim_best_model_spiking.pt`) | 86.45% | - | - |

Both leads fall inside one standard error of the runner-up (0.15 pts against
+-0.97; 0.35 against +-1.02), so the primary metric alone does not separate
them. The tie was broken on retiming to delay 100, where seed 2 wins in both
conditions. That lands on the same seed under both initializations, which is
also the cleaner matched pair.

Provenance: `training/exp1/{hippo,rand_complex}_s2/base/.../best_eval_2.pt`.

### Lap counting: `exp5_random` seeds, from `lapcount_sweep.json`

Evaluated at K = 2..16 laps, 150 episodes per cell. All three seeds are perfect
inside the trained range (K = 2..6), so extrapolation to unseen lap counts
separates them.

| seed | K=2-6 | K=7-16 | all K | VP (K=2-6) |
| --- | --- | --- | --- | --- |
| **3 (shipped)** | 100.00% | **91.80%** | **94.53%** | 0.9952 |
| 4 | 100.00% | 76.13% | 84.09% | 0.9990 |
| 2 | 100.00% | 68.60% | 79.07% | 0.9996 |

Provenance: `training/exp5_random/seed_3/best_eval.pt`.

This is the **redesigned** lap-counting task (`envs/lap_random.py`: randomised
lap durations, a three-symbol observation alphabet whose `[0,0]` EMPTY signal
drives the SSM with no input). It is not comparable to the superseded
fixed-30-step task -- checkpoints from the old design score 0% here and vice
versa, because the two environments assign different meanings to the same
2-D observation.

## Loading

```bash
python train_and_plot_3stim.py --spike --delay 30 --n_neurons 50 \
    --init_method hippo --load_model --model_path models/3stim_hippo_best.pt \
    --n_eval_episodes 2000
```

`--init_method rand_complex` and `models/3stim_rand_complex_best.pt` for the
other. The init only builds the network before the checkpoint overwrites it, so
it must match the architecture, not the weights.

`lap_counting_best.pt` is an 80-unit `AC_SSM_stack` for `Laps_Random`. Note that
`train_landmark_laps.py --mode grid` is defined for `--env landmark` only and
will refuse this checkpoint; the script that evaluates it across lap counts is
`eval_lapcount_sweep.py`, which is not carried in this repo.
