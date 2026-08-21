# Released checkpoints

| file | task | script | selected by |
| --- | --- | --- | --- |
| `3stim_hippo_best.pt` | 3-stim interval discrimination, delay 30 | `train_and_plot_3stim.py` | 89.45% |
| `3stim_rand_complex_best.pt` | same, random-complex init | `train_and_plot_3stim.py` | 88.45% |
| `lap_counting_best.pt` | lap counting under randomised timing | `train_landmark_laps.py` (`--env random`) | 94.53% mean over K=2..16 |

### 3-stim:

| candidate | clean acc | recorded | retimed d100 |
| --- | --- | --- | --- |
| **hippo** | **89.45%** | 89.0 | 82.3 |
| **rand_complex** | **88.45%** | 89.6 | 80.4 |

### Lap counting:

Evaluated at K = 2..16 laps, 150 episodes per cell. All three seeds are perfect
inside the trained range (K = 2..6), so extrapolation to unseen lap counts
separates them.

| seed | K=2-6 | K=7-16 | all K | VP (K=2-6) |
| --- | --- | --- | --- | --- |
| **3 (shipped)** | 100.00% | **91.80%** | **94.53%** | 0.9952 |
| 4 | 100.00% | 76.13% | 84.09% | 0.9990 |
| 2 | 100.00% | 68.60% | 79.07% | 0.9996 |

| K | verified (n=300) | sweep (n=150) | VP | false alarms/ep |
| --- | --- | --- | --- | --- |
| 2 | 100.00% | 100.00% | 0.9900 | 0.05 |
| 3 | 100.00% | 100.00% | 0.9957 | 0.03 |
| 4 | 100.00% | 100.00% | 0.9959 | 0.04 |
| 5 | 100.00% | 100.00% | 0.9974 | 0.03 |
| 6 | 100.00% | 100.00% | 0.9974 | 0.03 |
| 10 | 99.67% | 100.00% | 0.2306 | 70.68 |
| 13 | 91.67% | 90.67% | 0.1124 | 206.17 |
| 16 | 75.33% | 75.33% | 0.0804 | 361.04 |


## Loading

```bash
python train_and_plot_3stim.py --spike --delay 30 --n_neurons 50 \
    --init_method hippo --load_model --model_path models/3stim_hippo_best.pt \
    --n_eval_episodes 2000
```

`--init_method rand_complex` and `models/3stim_rand_complex_best.pt` for the
other. The init only builds the network before the checkpoint overwrites it, so
it must match the architecture, not the weights.

`lap_counting_best.pt` is an 80-unit `AC_SSM_stack` for `Laps_Random`:

```bash
python eval_lap_counting.py --ckpt models/lap_counting_best.pt \
    --k_min 2 --k_max 16 --n_episodes 300
```

`eval_lap_counting.py` defaults to the training distribution, so `K > 6`
measures extrapolation. Note that `train_landmark_laps.py --mode grid` is
defined for `--env landmark` only and will refuse this checkpoint.
