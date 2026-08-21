"""
Laps_Random -- random lap COUNT, random lap LENGTH, MID-LAP pauses, and every
episode padded to the same total length.

New env. envs/lap_counting.py and envs/lap_landmark.py are untouched: Exp 1, Exp 2,
Fig. 5 and the completed exp5 run all depend on those.

WHY THIS EXISTS
---------------
The exp5 analyses needed equal-frequency absolute-time "bands" to hold elapsed time
fixed while decoding lap number, because episodes had different lengths and lap
count was fixed at 4. That machinery was awkward: band widths, usable-band counts,
per-band chance levels. Three changes remove it:

  1. RANDOM LAP COUNT. K ~ U{k_min..k_max} per episode. Consequence beyond
     simplicity: `predicted_lap_count == true_lap_count` stops being degenerate.
     With K fixed at 4, any policy emitting exactly 4 times anywhere scored 100%
     count accuracy; now the count is a real task variable again.
  2. RANDOM LAP LENGTH, independently per lap, as before.
  3. FIXED TOTAL EPISODE LENGTH via padding. Every episode is exactly
     `total_steps` timesteps, so the collected states form a rectangular
     (episodes x T x units) array and lap identity can be decoded AT AN EXACT
     TIMESTEP t across episodes. That is perfect time matching by construction --
     no bands, no widths, no per-band chance bookkeeping.

MID-LAP PAUSES
--------------
Each lap consists of `d` RUNNING steps. Between 0 and `pause_count_range[1]` pauses
are inserted at random INTERIOR running-step boundaries of that lap, each of random
length. The landmark still fires on the lap's final running step, so a pause never
falls between a lap's end and its landmark.

This is different from the post-landmark pauses in Laps_Landmark, and the
difference matters. A post-landmark pause only shifts a lap in absolute time -- it
cannot spread lap 1's landmark time at all, since it lands after the landmark. A
mid-lap pause breaks the monotone mapping between within-lap POSITION and
within-lap ELAPSED TIME, and it spreads every lap including the first. Two phases
are therefore recorded per timestep and they are no longer the same quantity:
    phase_dist = running steps completed / d      (position along the track)
    phase_time = timesteps since lap onset / span (elapsed time within the lap)
`inter_lap_pause_range` additionally reinstates post-landmark pauses; it defaults
to off.

A PAUSE IS THE EMPTY SIGNAL [0, 0]
---------------------------------
The observation stays 2-D -- no new input channel, so input_dimensions=2 is
unchanged -- but a pause is not "running". It is the absence of any input:

    [1, 0]  running
    [0, 1]  landmark (one timestep)
    [0, 0]  paused, inter-lap dwell, or padding: nothing at all

This matters mechanistically, not just descriptively. With u = 0 the SSM update
loses its drive entirely and becomes h <- Lambda_bar h: pure autonomous decay. So a
pause is the strictest possible maintenance test -- lap identity must survive with
NO input, rather than merely with an uninformative constant input. It also makes a
pause genuinely distinguishable from slow running, which is what the reviewer's
"introduce random pauses" asks for and which the post-landmark pauses in
Laps_Landmark did not provide (there a pause was byte-identical to running, so it
was only a longer inter-landmark gap).

Padding at the end of the episode uses the same empty signal, for the same reason:
after the last lap, nothing is happening.
"""

import numpy as np
from gym import spaces

from envs.lap_landmark import landmark_reward

__all__ = ["Laps_Random"]

class Laps_Random(object):
    """
    Action space:
        0 = DO_NOTHING
        1 = EMIT (one pulse into the lap counter)

    Observation (2-D):
        [1, 0] = running
        [0, 1] = landmark (one timestep)
        [0, 0] = paused / inter-lap dwell / padding -- the EMPTY signal, no drive

    Drop-in for Laps_Landmark: reset() -> obs, step2(action) -> (obs, reward,
    task_stage), plus predicted_lap_count / true_lap_count / elapsed_t / lap_ends.
    """

    def __init__(self,
                 seed=1,
                 k_range=(2, 6),
                 lap_len_range=(15, 45),
                 pause_count_range=(0, 2),
                 pause_len_range=(0, 20),
                 inter_lap_pause_range=(0, 0),
                 total_steps=None,
                 hit_window=3,
                 hit_rwd=1.0,
                 miss_cost=1.0,
                 fa_cost=0.1,
                 term_rwd=2.0,
                 hold_extra=2):
        self.rng = np.random.RandomState(seed)
        self.k_range = (int(k_range[0]), int(k_range[1]))
        self.lap_len_range = (int(lap_len_range[0]), int(lap_len_range[1]))
        self.pause_count_range = (int(pause_count_range[0]), int(pause_count_range[1]))
        self.pause_len_range = (int(pause_len_range[0]), int(pause_len_range[1]))
        self.inter_lap_pause_range = (int(inter_lap_pause_range[0]),
                                      int(inter_lap_pause_range[1]))
        self.hit_window = int(hit_window)
        self.hit_rwd = float(hit_rwd)
        self.miss_cost = float(miss_cost)
        self.fa_cost = float(fa_cost)
        self.term_rwd = float(term_rwd)
        self.hold_extra = int(hold_extra)

        if self.lap_len_range[0] <= self.hit_window + 1:
            raise ValueError(
                f"min lap length {self.lap_len_range[0]} must exceed hit_window+1 "
                f"({self.hit_window + 1}) or response windows overlap")

        worst = self.k_range[1] * (self.lap_len_range[1]
                                   + self.pause_count_range[1] * self.pause_len_range[1]
                                   + self.inter_lap_pause_range[1])
        need = worst + self.hit_window + self.hold_extra + 1
        self.total_steps = int(need if total_steps is None else total_steps)
        if self.total_steps < need:
            raise ValueError(
                f"total_steps={self.total_steps} cannot hold the worst-case "
                f"schedule ({need}). Raise it or shrink the ranges.")

        self.obs_dim = 2
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                            shape=(self.obs_dim,), dtype=np.float32)
        self.reset()

    _OBS = ([1, 0], [0, 1], [0, 0])
    RUNNING, LANDMARK, EMPTY = 0, 1, 2

    def _build(self):
        """Explicit per-timestep timeline. Padding is everything after the tail."""
        lo, hi = self.lap_len_range
        pc_lo, pc_hi = self.pause_count_range
        pl_lo, pl_hi = self.pause_len_range
        il_lo, il_hi = self.inter_lap_pause_range
        T = self.total_steps

        K = int(self.rng.randint(self.k_range[0], self.k_range[1] + 1))

        lap_of = np.full(T, -1, dtype=np.int64)
        is_pause = np.zeros(T, dtype=np.int64)
        is_cue = np.zeros(T, dtype=np.int64)
        run_done = np.full(T, -1, dtype=np.int64)
        obs_code = np.full(T, self.EMPTY, dtype=np.int64)
        lap_ends, lap_starts, lap_durations, lap_spans = [], [], [], []

        t = 0
        for k in range(K):
            d = int(self.rng.randint(lo, hi + 1))
            n_p = int(self.rng.randint(pc_lo, pc_hi + 1))
            if n_p > 0 and d > 1:
                pos = set(self.rng.choice(np.arange(1, d),
                                          size=min(n_p, d - 1), replace=False).tolist())
            else:
                pos = set()
            lap_starts.append(t)
            lap_durations.append(d)
            for s in range(1, d + 1):
                lap_of[t] = k
                run_done[t] = s
                is_cue[t] = 1 if s == d else 0
                obs_code[t] = self.LANDMARK if s == d else self.RUNNING
                t += 1
                if s in pos and pl_hi > 0:
                    for _ in range(int(self.rng.randint(pl_lo, pl_hi + 1))):
                        lap_of[t] = k
                        run_done[t] = s
                        is_pause[t] = 1
                        obs_code[t] = self.EMPTY
                        t += 1
            lap_ends.append(t - 1)
            lap_spans.append(t - lap_starts[-1])
            if il_hi > 0:
                for _ in range(int(self.rng.randint(il_lo, il_hi + 1))):
                    lap_of[t] = -1
                    is_pause[t] = 1
                    obs_code[t] = self.EMPTY
                    t += 1

        self.lap_count = K
        self.lap_ends = lap_ends
        self.lap_start_times = lap_starts
        self.lap_durations = lap_durations
        self.lap_spans = lap_spans
        self.lap_mid_times = [s + sp // 2 for s, sp in zip(lap_starts, lap_spans)]
        self.schedule_len = t
        self._lap_of = lap_of
        self._is_pause = is_pause
        self._is_cue = is_cue
        self._run_done = run_done
        self._obs_code = obs_code

    def reset(self):
        self._build()
        self.elapsed_t = 0
        self.true_lap_count = 0
        self.predicted_lap_count = 0
        self.task_stage = 'running'
        self.hold = 0
        self._win_open = False
        self._win_t0 = None
        self._win_hit = False
        self.press_times, self.hit_times, self.hit_lags, self.fa_times = [], [], [], []
        self.n_hit = self.n_miss = self.n_fa = self.n_extra = 0
        self.observation = list(self._OBS[self._obs_code[0]])
        return self.observation

    def step2(self, action):
        if self.task_stage == 'done':
            return self.observation, 0.0, self.task_stage

        t = self.elapsed_t
        reward = 0.0

        if self._is_cue[t]:
            self._win_open = True
            self._win_t0 = t
            self._win_hit = False
            self.true_lap_count += 1

        r, outcome = landmark_reward(
            int(self._is_cue[t]), int(action), self._win_open, self._win_hit,
            hit_rwd=self.hit_rwd, miss_cost=self.miss_cost, fa_cost=self.fa_cost)
        reward += r

        if action == 1:
            self.press_times.append(t)
        if outcome == 'hit':
            self._win_hit = True
            self.predicted_lap_count += 1
            self.n_hit += 1
            self.hit_times.append(t)
            self.hit_lags.append(t - self._win_t0)
        elif outcome == 'extra':
            self.n_extra += 1
        elif outcome == 'false_alarm':
            self.n_fa += 1
            self.fa_times.append(t)

        if self._win_open and t >= self._win_t0 + self.hit_window:
            if not self._win_hit:
                reward -= self.miss_cost
                self.n_miss += 1
            self._win_open = False

        self.elapsed_t += 1
        if self.elapsed_t >= self.total_steps:
            reward += (self.term_rwd
                       if self.predicted_lap_count == self.true_lap_count
                       else -self.term_rwd)
            self.observation = list(self._OBS[self.EMPTY])
            self.task_stage = 'done'
        else:
            nt = self.elapsed_t
            self.observation = list(self._OBS[self._obs_code[nt]])
            self.task_stage = 'lap_end' if self._is_cue[nt] else 'running'

        return self.observation, reward, self.task_stage

    def lap_index_at(self, t):
        """Lap this timestep belongs to, or -1 for padding / inter-lap dwell."""
        return int(self._lap_of[t])

    def is_pause_at(self, t):
        return bool(self._is_pause[t])

    def is_padding_at(self, t):
        return bool(self._lap_of[t] == -1 and not self._is_pause[t])

    def phase_dist_at(self, t):
        """Position along the lap: running steps completed / lap duration."""
        k = int(self._lap_of[t])
        if k < 0:
            return float('nan')
        return float(self._run_done[t]) / float(self.lap_durations[k])

    def phase_time_at(self, t):
        """Elapsed time within the lap / lap wall-clock span. Differs from
        phase_dist whenever the lap contains a mid-lap pause."""
        k = int(self._lap_of[t])
        if k < 0:
            return float('nan')
        return float(t - self.lap_start_times[k]) / float(self.lap_spans[k])

    def episode_summary(self):
        return {
            "true_count": int(self.true_lap_count),
            "pred_count": int(self.predicted_lap_count),
            "correct": int(self.predicted_lap_count == self.true_lap_count),
            "n_hit": int(self.n_hit), "n_miss": int(self.n_miss),
            "n_fa": int(self.n_fa), "n_extra": int(self.n_extra),
            "press_times": list(self.press_times),
            "hit_lags": list(self.hit_lags),
            "fa_times": list(self.fa_times),
            "lap_ends": list(self.lap_ends),
            "lap_durations": list(self.lap_durations),
            "lap_spans": list(self.lap_spans),
            "schedule_len": int(self.schedule_len),
            "total_steps": int(self.total_steps),
            "episode_len": int(self.total_steps),
            "n_pause_steps": int(self._is_pause.sum()),
        }

    def seed(self, seed=None):
        self.rng.seed(seed)
        return [seed]

    def render(self):
        t = min(self.elapsed_t, self.total_steps - 1)
        print(f"t={self.elapsed_t}/{self.total_steps} stage={self.task_stage} "
              f"lap={self.lap_index_at(t)}/{self.lap_count} "
              f"pause={self.is_pause_at(t)} pad={self.is_padding_at(t)} "
              f"true={self.true_lap_count} pred={self.predicted_lap_count}")
