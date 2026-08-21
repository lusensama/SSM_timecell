
import numpy as np
from gym import spaces

def landmark_reward(cue, action, win_open, win_hit,
                    hit_rwd=1.0, miss_cost=1.0, fa_cost=0.1):
    """
    Per-timestep reward. Deliberately has NO access to absolute time or to the
    landmark schedule -- see the module docstring. Do not add such arguments.

    Args:
        cue:      1 if a landmark cue was just observed by the agent, else 0.
                  (unused directly: a cue opens the window before this is called,
                  so its effect is carried by win_open. Kept in the signature to
                  make the "observation only" contract explicit.)
        action:   1 if the agent emitted this timestep, else 0.
        win_open: whether a response window is currently open.
        win_hit:  whether that window has already been credited with a hit.

    Returns:
        (reward, outcome) where outcome is one of
        'hit' | 'extra' | 'false_alarm' | 'none'.
        The 'miss' outcome is emitted by the caller at window expiry, since it is
        the absence of an action rather than an action.
    """
    del cue
    if action != 1:
        return 0.0, 'none'
    if win_open and not win_hit:
        return float(hit_rwd), 'hit'
    if win_open:
        return -float(fa_cost), 'extra'
    return -float(fa_cost), 'false_alarm'

class Laps_Landmark(object):
    """
    Action space:
        0 = DO_NOTHING
        1 = EMIT (one pulse into the single-neuron lap counter)

    Observation (2-D, same as Laps_Counting so the model needs no change):
        [1, 0] = running
        [0, 1] = landmark cue (one timestep)

    Interface is a drop-in for Laps_Counting: reset() -> obs,
    step2(action) -> (obs, reward, task_stage), and the attributes
    predicted_lap_count / true_lap_count / elapsed_t / lap_ends / task_stage.
    """

    def __init__(self,
                 seed=1,
                 fixed_laps=4,
                 vary_lap_len=True,
                 lap_length=30,
                 lap_len_range=(20, 45),
                 pause_range=(0, 15),
                 hit_window=3,
                 hit_rwd=1.0,
                 miss_cost=1.0,
                 fa_cost=0.1,
                 term_rwd=2.0,
                 hold_extra=2):
        """
        vary_lap_len: draw each lap's duration independently from lap_len_range.
            When False, every lap is exactly `lap_length` steps and no pauses are
            inserted -- the old fixed-pacing regime, kept for the eval grid.
        pause_range: extra plain-running dwell after a lap's landmark, delaying
            the next lap. Only used when vary_lap_len is True.
        hold_extra: dwell steps after the last window closes, so the final lap's
            window can complete before the episode ends.
        """
        self.rng = np.random.RandomState(seed)
        self.base_lap_count = fixed_laps
        self.vary_lap_len = vary_lap_len
        self.lap_len = int(lap_length)
        self.lap_len_range = (int(lap_len_range[0]), int(lap_len_range[1]))
        self.pause_range = (int(pause_range[0]), int(pause_range[1]))
        self.hit_window = int(hit_window)
        self.hit_rwd = float(hit_rwd)
        self.miss_cost = float(miss_cost)
        self.fa_cost = float(fa_cost)
        self.term_rwd = float(term_rwd)
        self.hold_extra = int(hold_extra)

        if self.vary_lap_len and self.lap_len_range[0] <= self.hit_window + 1:
            raise ValueError(
                f"min lap length {self.lap_len_range[0]} must exceed "
                f"hit_window+1 ({self.hit_window + 1}) or response windows overlap")

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,),
                                            dtype=np.float32)
        self.reset()

    def _draw_schedule(self):
        """Landmark times for one episode, plus per-lap metadata."""
        lap_ends, lap_starts, lap_durations = [], [], []
        start = 0
        lo, hi = self.lap_len_range
        p_lo, p_hi = self.pause_range
        for _ in range(self.lap_count):
            if self.vary_lap_len:
                d = int(self.rng.randint(lo, hi + 1))
                pause = int(self.rng.randint(p_lo, p_hi + 1)) if p_hi > 0 else 0
            else:
                d, pause = self.lap_len, 0
            lap_starts.append(start)
            lap_durations.append(d)
            lap_ends.append(start + d)
            start = start + d + pause
        return lap_ends, lap_starts, lap_durations

    def reset(self):
        self.lap_count = self.base_lap_count
        self.lap_ends, self.lap_start_times, self.lap_durations = self._draw_schedule()
        self.lap_mid_times = [s + d // 2 for s, d in
                              zip(self.lap_start_times, self.lap_durations)]
        self.lap_end_times = list(self.lap_ends)
        self._cue_set = set(self.lap_ends)
        self.T_end = self.lap_ends[-1] + self.hit_window + self.hold_extra

        self.elapsed_t = 0
        self.true_lap_count = 0
        self.predicted_lap_count = 0
        self.task_stage = 'running'
        self.hold = 0

        self._win_open = False
        self._win_t0 = None
        self._win_hit = False

        self.press_times = []
        self.hit_times = []
        self.hit_lags = []
        self.fa_times = []
        self.n_hit = 0
        self.n_miss = 0
        self.n_fa = 0
        self.n_extra = 0

        self._cue_now = self.elapsed_t in self._cue_set
        self.observation = [0, 1] if self._cue_now else [1, 0]
        return self.observation

    def step2(self, action):
        """
        `action` is the response to the observation the agent currently holds,
        i.e. to the cue state at self.elapsed_t.
        """
        if self.task_stage == 'done':
            return self.observation, 0.0, self.task_stage

        t = self.elapsed_t
        reward = 0.0

        if self._cue_now:
            self._win_open = True
            self._win_t0 = t
            self._win_hit = False
            self.true_lap_count += 1

        r, outcome = landmark_reward(
            int(self._cue_now), int(action), self._win_open, self._win_hit,
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
        if self.elapsed_t > self.T_end:
            reward += (self.term_rwd
                       if self.predicted_lap_count == self.true_lap_count
                       else -self.term_rwd)
            self._cue_now = False
            self.observation = [1, 0]
            self.task_stage = 'done'
        else:
            self._cue_now = self.elapsed_t in self._cue_set
            self.observation = [0, 1] if self._cue_now else [1, 0]
            self.task_stage = 'lap_end' if self._cue_now else 'running'

        return self.observation, reward, self.task_stage

    def seed(self, seed=None):
        self.rng.seed(seed)
        return [seed]

    def episode_summary(self):
        """Per-episode behavioral record, consumed by the eval script."""
        return {
            "true_count": int(self.true_lap_count),
            "pred_count": int(self.predicted_lap_count),
            "correct": int(self.predicted_lap_count == self.true_lap_count),
            "n_hit": int(self.n_hit),
            "n_miss": int(self.n_miss),
            "n_fa": int(self.n_fa),
            "n_extra": int(self.n_extra),
            "press_times": list(self.press_times),
            "hit_lags": list(self.hit_lags),
            "fa_times": list(self.fa_times),
            "lap_ends": list(self.lap_ends),
            "lap_durations": list(self.lap_durations),
            "episode_len": int(self.T_end + 1),
        }

    def render(self):
        print(f"t={self.elapsed_t} stage={self.task_stage} "
              f"laps={self.true_lap_count}/{self.lap_count} "
              f"pred={self.predicted_lap_count} obs={self.observation} "
              f"win_open={self._win_open} win_hit={self._win_hit}")

