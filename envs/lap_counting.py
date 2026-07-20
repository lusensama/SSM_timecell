import numpy as np
import random
import copy
from gym import spaces
from typing import List,Iterable, Callable, Optional,Union
import math

def generate_lap_end_times(
        lap_count: int,
        lap_length: int,
        *,                    # keyword-only from here on
        jitter: float = 1.0,  # σ of the Gaussian noise
         seed: Optional[int] = None,
        as_int: bool = True
) -> List[Union[int, float]]:
    """
    Return *exactly* `lap_count` timestamps—one drawn around each nominal
    lap end k × lap_length (k = 1 … lap_count).

    • jitter:  standard deviation of the noise added to each lap end
               (use 0 for deterministic ends)
    • seed:    set for reproducible output
    • as_int:  round to nearest int (default) or keep floats
    """
    if seed is not None:
        random.seed(seed)

    max_time = lap_count * lap_length + 1          # absolute ceiling
    half_window = lap_length / 2                   # clamp band

    times = []
    for k in range(1, lap_count + 1):
        center = k * lap_length
        t = random.gauss(center, jitter)           # add noise

        # keep it within half a lap on either side of the centre
        t = max(center - half_window, min(t, center + half_window))
        # enforce the global ceiling
        t = min(t, max_time)

        times.append(round(t) if as_int else t)

    return times
def spike_penalty_reward(scale: float = 1.0,
                         pos_value: float = 1.0) -> Callable[[int], float]:
    """
    Returns a reward‑shaping function f(d) with:

        d = 0  →  +pos_value                (exact hit)
        d > 0 →  -(1 - exp(-d / scale))     (negative, steeper as d grows)

    • `scale`  controls how quickly the penalty saturates near –1.
    • `pos_value` lets you set the exact‑hit bonus (default +1).
    """
    neg_curve = lambda d: -(1 - math.exp(-d / scale))
    return lambda d: pos_value if d == 0 else neg_curve(d)
def tanh_spike(scale: float = 1.0,
               pos_value: float = 1.0) -> Callable[[int], float]:
    return lambda d: pos_value if d == 0 else -math.tanh(d / scale)
def timing_reward(
    current_step: int,
    timings: List[int],
    gap_to_reward: Callable[[int], float],
) -> float:
    if not timings:
        raise ValueError("`timings` must contain at least one element.")
    nearest_gap = min(abs(current_step - t) for t in timings)
    return gap_to_reward(nearest_gap)

class Laps_Counting(object):
    """
    An environment that generates a stream of 0s and 1s to simulate laps.
    The goal for the agent is to correctly count the number of laps.

    Action Space:
        - 0: DO_NOTHING
        - 1: REPORT_LAP

    Observations:
        - 0: In the middle of a lap or trial has ended.
        - 1: End of a lap signal (lasts for one timestep).

    Reward Logic:
        - 'running': -0.1 for reporting a lap (action 1), 0 otherwise.
        - 'lap_end': +1 for correctly reporting the lap (action 1), -1 otherwise.
        - 'trial_end': A final reward calculated as:
                       - (abs(true_laps - predicted_laps) ** 2)
                       This penalizes inaccurate final counts quadratically.
    """

    def __init__(self,
                 final_rwd=10, # Note: final_rwd is kept for compatibility but not used in the new reward scheme
                 seed=1,
                 lap_length=10,
                 fixed_laps=4,
                 approx=False,
                 ext_lap = -1,
                 jitter=1.0,
                 rand_ext=False,
                 randomize_laps=False,
                 eval_hold=False):
        """
        Initializes the laps counting environment.
        """
        # Random generator for lap length variability
        self.rng = np.random.RandomState(seed)
        self.elapsed_t = 0
        self.base_lap_count = fixed_laps
        self.randomize_laps = randomize_laps
        self.lap_count = fixed_laps  # Will be set in reset()
        self.lap_len = lap_length
        self.approx=approx
        # Define action and observation spaces for Gym compatibility
        self.action_space = spaces.Discrete(2)
        self.final_rwd = final_rwd
        # Initialize the state of the environment
        self.ext_lap = ext_lap
        self.jitter = jitter
        self.reset()
        self.rand_ext = rand_ext
        self.eval_hold = eval_hold

    def reset(self):
        """
        Resets the environment to the starting state for a new episode.
        """
        # Randomize lap count if enabled
        if getattr(self, 'randomize_laps', False):
            self.lap_count = self.rng.randint(2, self.base_lap_count)
        else:
            self.lap_count = self.base_lap_count
        self.task_stage = 'running'
        self.true_lap_count = 0
        self.predicted_lap_count = 0
        self.steps_in_current_lap = 0
        self.current_lap_index = 0
        self.current_lap_length = self.lap_len
        self.observation = [1, 0]
        self.reward = 0.0
        self.elapsed_t = 0
        self.hold=0
        self.reward_fn = spike_penalty_reward(scale=2.0, pos_value=1.0)
        self.rand_ext =False
        if self.approx:
            self.lap_ends=generate_lap_end_times(self.lap_count, self.lap_len, jitter=self.jitter)
        else:
            self.lap_ends=[i for i in range(self.lap_len, self.lap_count*self.lap_len+1, self.lap_len)]
        if self.rand_ext:
            self.ext_lap = np.random.randint(0, self.lap_count)
        if self.ext_lap >= 0:
            self.lap_ends = [self.lap_ends[i]+self.lap_len//2 for i in range(self.ext_lap, len(self.lap_ends))]
        self.rng.seed(np.random.randint(0, 2**32 - 1))
        return self.observation

    def step2(self, action):
        """
        Advances the environment by one timestep based on the agent's action.

        Args:
            action (int): The action taken by the agent (0 or 1).

        Returns:
            tuple: A tuple containing:
                - observation (int): The observation for the next state.
                - reward (float): The reward received from the action.
                - task_stage (str): The current stage of the task ('running', 'lap_end', 'trial_end', 'done').
        """
        # If the trial has ended, do nothing further
        if self.task_stage == 'done':
            return self.observation, self.reward, self.task_stage

        # --- State: running ---
        if self.task_stage == 'running':
            if action ==1:
                self.predicted_lap_count += 1
                self.reward = timing_reward(self.elapsed_t, self.lap_ends, self.reward_fn)*3
            else:
                self.reward = 0.1
            if self.elapsed_t+1 in self.lap_ends:
                self.task_stage = 'lap_end'
                self.observation = [0, 1] # Signal lap end
            else:
                self.observation = [1, 0] # Still in the middle of a lap

        # --- State: lap_end ---
        elif self.task_stage == 'lap_end':
            if action == 1:
                self.reward = timing_reward(self.elapsed_t, self.lap_ends, self.reward_fn)*1
                # self.reward = 0
                self.predicted_lap_count += 1
            else:
                # self.reward = timing_reward(self.elapsed_t, self.lap_ends, self.reward_fn)*20
                self.reward = -1
            self.true_lap_count += 1

            # Check if this was the final lap
            if self.true_lap_count >= len(self.lap_ends):
                self.task_stage = 'trial_end'
            else:
                self.task_stage = 'running'
                self.observation = [0, 1]  # Reset observation for the start of the new lap

        # --- State: trial_end ---
        if self.task_stage == 'trial_end':
            # Calculate the final disproportionate reward
            count_difference = abs(self.true_lap_count - self.predicted_lap_count)
            self.reward = self.final_rwd if count_difference ==0 else -self.final_rwd
            self.reward *=1
            # self.reward = -(np.abs(count_difference))
            if self.eval_hold and self.hold < 15:
                self.hold += 1
                self.observation = [1, 0]  # Trial is over
            else:
                self.observation = [1,0]  # Trial is over
                # if self.hold>=self.lap_len:
                self.task_stage = 'done' # Mark as done
        self.elapsed_t +=1
        return self.observation, self.reward, self.task_stage


    def update_lap_configs(self, num_laps, lap_length):
        """
        Updates the lap configuration for the environment.

        Args:
            num_laps (int): The number of laps for subsequent episodes.
            lap_length (int or None): The length of each lap. If None, uses random lengths.
        """
        lap_configs = []
        if self.fixed_length_default:
            lap_configs = [(num_laps, lap_length)]
        else:
            for _ in range(num_laps):
                rand_len = self.rng.randint(3, lap_length + 1) if lap_length > 3 else 3
                lap_configs.append((1, rand_len))

        self.lap_lengths = []
        for n, length in lap_configs:
            self.lap_lengths.extend([length] * n)

        self.total_laps_in_episode = (
            min(self.max_laps_default, len(self.lap_lengths))
            if self.max_laps_default is not None else len(self.lap_lengths)
        )
        if self.total_laps_in_episode == 0:
            raise ValueError("No laps configured. Check lap_length or num_laps.")
        self.reset()

    def seed(self, seed=None):
        """
        Reseed the internal random number generator.
        """
        self.rng.seed(seed)
        return [seed]

    def render(self):
        """
        Displays the current internal state.
        """
        print(
            f"Stage: {self.task_stage} | Lap: {self.true_lap_count}/"
            f"{self.total_laps_in_episode} | Step: {self.steps_in_current_lap}/"
            f"{self.current_lap_length} | Obs: {self.observation} | "
            f"Last Reward: {self.reward}"
        )
