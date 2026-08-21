import numpy as np
from numpy import array
import random
import copy
from gym import spaces

class IntervalDiscrimination(object):
    '''
    Interval discrimination, head-fixed agent. After an initiation cue, the agent pokes to initiate the task.
    Then two stimulus are shown sequentially, seperated by a delay period. The duration of each stimulus is
    randomly drawn from [10,15,20,25,30,35,40]. The length of the delay period is 20 time steps. After the second
    stimulus presentation, a "Go" cue would show up, and the agent is going to produce an action of either "0"
    or "1" to indicate whether the first stimulus was longer in duration or the second one.
    '''

    def __init__(self, rwd=10, inc_rwd=-10, seed=1):
        self.stimulus_set = [10,15, 20,25, 30,35, 40,45, 50]
        self.delay_duration = 30
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.MultiBinary(2)
        self.rng = np.random.RandomState(seed)
        self.reward = 0
        self.task_stage = 'init'
        self.done = False
        self.first_stim = np.random.choice(self.stimulus_set)
        self.second_stim = self.select_second_stim()
        self.rwd = rwd
        self.inc_rwd = inc_rwd
        self.elapsed_t = 0
        self.correct_trial = False
        self.observation = [1,1]
        self.groundtruth = self.first_stim > self.second_stim

    def reset(self):
        self.reward = 0
        self.task_stage = 'init'
        self.done = False
        self.first_stim = np.random.choice(self.stimulus_set)
        self.second_stim = self.select_second_stim()
        self.groundtruth = self.first_stim > self.second_stim
        self.elapsed_t = 0
        self.correct_trial = False
        self.observation = [1,1]

    def step(self, action=None):
        """
        :param action
        :return: observation, reward, done, info
        """

        if self.task_stage == "init":
            if action == 1:
                self.task_stage = "first_stim"
                self.observation = [1,0]
                self.reward = 1
            else:
                self.reward = -1

        elif self.task_stage == "first_stim":
            if self.elapsed_t >= self.first_stim:
                self.task_stage = "delay_init"
                self.observation = [1,1]
                self.elapsed_t = 0
            else:
                self.elapsed_t += 1
        elif self.task_stage == "delay_init":
            self.task_stage = "delay"
            self.observation = [0,0]
        elif self.task_stage == "delay":
            if self.elapsed_t >= self.delay_duration:
                self.task_stage = "delay_end"
                self.observation = [1,1]
                self.elapsed_t = 0
            else:
                self.elapsed_t += 1
        elif self.task_stage == "delay_end":
            self.task_stage = "second_stim"
            self.observation = [0,1]
        elif self.task_stage == "second_stim":
            if self.elapsed_t >= self.second_stim:
                self.task_stage = "choice_init"
                self.observation = [1,1]
                self.elapsed_t = 0
            else:
                self.elapsed_t += 1

        elif self.task_stage == "choice_init":
            if action == self.groundtruth:
                self.reward = self.rwd
                self.correct_trial = True
            else:
                self.reward = self.inc_rwd
            self.done = True

        return self.observation, self.reward, self.done
    def calc_reward_without_stepping(self, action=None):
        """
        :param action
        :return: observation, reward, done
        """

        if self.task_stage == "init":
            if action == 1:
                reward = 1
            else:
                reward = -1
        elif self.task_stage == "choice_init":
            if action == self.groundtruth:
                reward = self.rwd
            else:
                reward = self.inc_rwd
        else:
            reward = 0

        return reward

    def select_second_stim(self):
        stimulus_set_copy = copy.deepcopy(self.stimulus_set)
        stimulus_set_copy.remove(self.first_stim)
        return np.random.choice(stimulus_set_copy)

class IntervalDiscrimination3(object):
    '''
    Interval discrimination, head-fixed agent with three stimuli.
    After an initiation cue, the agent pokes to begin. Then three stimuli are
    shown sequentially, each lasting a randomly drawn duration from stimulus_set,
    separated by a fixed delay. After the third stimulus, a "Go" cue appears and
    the agent must choose which stimulus was longest (action in {0,1,2}).
    '''

    def __init__(self, rwd=10, inc_rwd=-10, seed=1):
        self.stimulus_set   = [10,20,30,]
        self.delay_duration = 50

        self.action_space      = spaces.Discrete(3)
        self.observation_space = spaces.MultiBinary(3)

        self.rng     = np.random.RandomState(seed)
        self.rwd     = rwd
        self.inc_rwd = inc_rwd

        self.first_stim  = self.rng.choice(self.stimulus_set)
        self.second_stim = self.select_second_stim()
        self.third_stim  = self.select_third_stim()

        self.groundtruth = int(np.argmax([self.first_stim,
                                          self.second_stim,
                                          self.third_stim]))

        self.reset()

    def reset(self):
        self.reward       = 0
        self.task_stage   = 'init'
        self.done         = False
        self.elapsed_t    = 0
        self.correct_trial = False

        self.first_stim  = self.rng.choice(self.stimulus_set)
        self.second_stim = self.select_second_stim()
        self.third_stim  = self.select_third_stim()

        self.groundtruth = int(np.argmax([self.first_stim,
                                          self.second_stim,
                                          self.third_stim]))

        self.observation = [1,1,1]
        return self.observation

    def step(self, action=None):
        obs, rew, done = None, 0, False

        if self.task_stage == "init":
            if action == 1:
                self.task_stage = "first_stim"
                self.observation = [1,0,0]
                self.reward = 1
            else:
                self.reward = -1

        elif self.task_stage == "first_stim":
            if self.elapsed_t >= self.first_stim:
                self.task_stage = "delay_init"
                self.observation = [1,1,1]
                self.elapsed_t = 0
            else:
                self.elapsed_t += 1

        elif self.task_stage == "delay_init":
            self.task_stage = "delay"
            self.observation = [0,0,0]

        elif self.task_stage == "delay":
            if self.elapsed_t >= self.delay_duration:
                self.task_stage = "delay_end"
                self.observation = [1,1,1]
                self.elapsed_t = 0
            else:
                self.elapsed_t += 1

        elif self.task_stage == "delay_end":
            self.task_stage = "second_stim"
            self.observation = [0,1,0]

        elif self.task_stage == "second_stim":
            if self.elapsed_t >= self.second_stim:
                self.task_stage = "second_delay_init"
                self.observation = [1,1,1]
                self.elapsed_t = 0
            else:
                self.elapsed_t += 1

        elif self.task_stage == "second_delay_init":
            self.task_stage = "second_delay"
            self.observation = [0,0,0]

        elif self.task_stage == "second_delay":
            if self.elapsed_t >= self.delay_duration:
                self.task_stage = "second_delay_end"
                self.observation = [1,1,1]
                self.elapsed_t = 0
            else:
                self.elapsed_t += 1

        elif self.task_stage == "second_delay_end":
            self.task_stage = "third_stim"
            self.observation = [0,0,1]

        elif self.task_stage == "third_stim":
            if self.elapsed_t >= self.third_stim:
                self.task_stage = "choice_init"
                self.observation = [1,1,1]
                self.elapsed_t = 0
            else:
                self.elapsed_t += 1

        elif self.task_stage == "choice_init":
            if action == self.groundtruth:
                self.reward = self.rwd
                self.correct_trial = True
            else:
                self.reward = self.inc_rwd
            self.done = True

        return self.observation, self.reward, self.done

    def calc_reward_without_stepping(self, action=None):
        if self.task_stage == "init":
            return 1 if action == 1 else -1
        elif self.task_stage == "choice_init":
            return self.rwd if action == self.groundtruth else self.inc_rwd
        else:
            return 0

    def select_second_stim(self):
        choices = copy.deepcopy(self.stimulus_set)
        choices.remove(self.first_stim)
        return self.rng.choice(choices)

    def select_third_stim(self):
        choices = copy.deepcopy(self.stimulus_set)
        choices.remove(self.first_stim)
        choices.remove(self.second_stim)
        return self.rng.choice(choices)

class IntDiscrim3_Intermediate(object):
    '''
    Interval discrimination, head-fixed agent with three stimuli.
    After an initiation cue, the agent pokes to begin. Then three stimuli are
    shown sequentially, each lasting a randomly drawn duration from stimulus_set,
    separated by a fixed delay. After the third stimulus, a "Go" cue appears and
    the agent must choose which stimulus was longest (action in {0,1,2}).
    '''

    def __init__(self, rwd=10, inc_rwd=-10, seed=1, delay=100, fixed_delay=True):
        self.stimulus_set   = [10,15, 20,25, 30,35, 40,45,50]
        multiplier = 1
        self.stimulus_set = [int(d * multiplier) for d in self.stimulus_set]
        if fixed_delay:
            self.delay_set = [delay]
        else:
            self.delay_set = list(range(10, delay, 10))
        self.action_space      = spaces.Discrete(3)
        self.observation_space = spaces.MultiBinary(3)

        self.rng     = np.random.RandomState(seed)
        self.rwd     = rwd
        self.inc_rwd = inc_rwd

        delay_c = self.rng.choice(self.delay_set)
        self.delay_duration = int(delay_c * multiplier)
        self.first_stim  = self.rng.choice(self.stimulus_set)
        self.second_stim = self.select_second_stim()
        self.third_stim  = self.select_third_stim()

        self.groundtruth = int(np.argmax([self.first_stim,
                                          self.second_stim,
                                          self.third_stim]))
        self.reset()

    def reset(self):
        self.rng.seed(np.random.randint(0, 2**32 - 1))
        self.reward       = 0
        self.task_stage   = 'init'
        self.done         = False
        self.elapsed_t    = 0
        self.correct_trial = False

        self.first_stim  = self.rng.choice(self.stimulus_set)
        self.second_stim = self.select_second_stim()
        self.third_stim  = self.select_third_stim()

        self.groundtruth = int(np.argmax([self.first_stim,
                                          self.second_stim,
                                          self.third_stim]))
        self.delay_duration1 = self.rng.choice(self.delay_set)
        self.delay_duration2 = self.rng.choice(self.delay_set)
        self.observation = [1,1,1]
        return self.observation

    def step(self, action=None):
        obs, rew, done = None, 0, False

        if self.task_stage == "init":
            if action == 1:
                self.task_stage = "first_stim"
                self.observation = [1,0,0]
                self.reward = 1
            else:
                self.reward = -1

        elif self.task_stage == "first_stim":
            if self.elapsed_t >= self.first_stim:
                self.task_stage = "delay_init"
                self.observation = [1,1,1]
                self.elapsed_t = 0
            else:
                self.elapsed_t += 1

        elif self.task_stage == "delay_init":
            self.task_stage = "delay"
            self.observation = [0,0,0]

        elif self.task_stage == "delay":
            if self.elapsed_t >= self.delay_duration1:
                self.task_stage = "delay_end"
                self.observation = [1,1,1]
                self.elapsed_t = 0
            else:
                self.elapsed_t += 1

        elif self.task_stage == "delay_end":
            self.task_stage = "second_stim"
            self.observation = [0,1,0]

        elif self.task_stage == "second_stim":
            if self.elapsed_t >= self.second_stim:
                self.task_stage = "intermediate_choice_init"
                self.observation = [1, 1, 1]
                self.elapsed_t = 0
            else:
                self.elapsed_t += 1
        elif self.task_stage == "intermediate_choice_init":
            intermediate_groundtruth = int((self.first_stim < self.second_stim))

            if action is not None:
                if action == intermediate_groundtruth:
                    self.reward = self.rwd * 0.5
                    self.correct_intermediate_trial = True
                else:
                    self.reward = self.inc_rwd * 0.5
                    self.correct_intermediate_trial = False
                self.task_stage = "second_delay_init"
                self.observation = [0, 0, 0]
                self.elapsed_t = 0
            else:
                self.reward = self.inc_rwd * 0.1

        elif self.task_stage == "second_delay_init":
            self.task_stage = "second_delay"
            self.observation = [0,0,0]

        elif self.task_stage == "second_delay":
            if self.elapsed_t >= self.delay_duration2:
                self.task_stage = "second_delay_end"
                self.observation = [1,1,1]
                self.elapsed_t = 0
            else:
                self.elapsed_t += 1

        elif self.task_stage == "second_delay_end":
            self.task_stage = "third_stim"
            self.observation = [0,0,1]

        elif self.task_stage == "third_stim":
            if self.elapsed_t >= self.third_stim:
                self.task_stage = "choice_init"
                self.observation = [1,1,1]
                self.elapsed_t = 0
            else:
                self.elapsed_t += 1

        elif self.task_stage == "choice_init":
            if action == self.groundtruth:
                self.reward = self.rwd
                self.correct_trial = True
            else:
                self.reward = self.inc_rwd
            self.done = True

        return self.observation, self.reward, self.done

    def calc_reward_without_stepping(self, action=None):
        if self.task_stage == "init":
            return 1 if action == 1 else -1
        elif self.task_stage == "choice_init":
            return self.rwd if action == self.groundtruth else self.inc_rwd
        else:
            return 0

    def select_second_stim(self):
        choices = copy.deepcopy(self.stimulus_set)
        choices.remove(self.first_stim)
        return self.rng.choice(choices)

    def select_third_stim(self):
        choices = copy.deepcopy(self.stimulus_set)
        choices.remove(self.first_stim)
        choices.remove(self.second_stim)
        return self.rng.choice(choices)

class IntDiscrim3_Intermediate_Pre_delay(object):
    '''
    Interval discrimination, head-fixed agent with three stimuli.
    After an initiation cue, the agent pokes to begin. Then three stimuli are
    shown sequentially, each lasting a randomly drawn duration from stimulus_set,
    separated by a fixed delay. After the third stimulus, a "Go" cue appears and
    the agent must choose which stimulus was longest (action in {0,1,2}).
    '''

    def __init__(self, rwd=10, inc_rwd=-10, seed=1, delay=100, max_pre=50):
        self.stimulus_set   = [10,15, 20,25, 30,35, 40,45, 50]
        self.pre_delay_set = list(range(max_pre))
        self.delay_duration = delay

        self.action_space      = spaces.Discrete(3)
        self.observation_space = spaces.MultiBinary(3)

        self.rng     = np.random.RandomState(seed)
        self.rwd     = rwd
        self.inc_rwd = inc_rwd

        self.pre_delay = self.rng.choice(self.pre_delay_set)
        self.first_stim  = self.rng.choice(self.stimulus_set)
        self.second_stim = self.select_second_stim()
        self.third_stim  = self.select_third_stim()

        self.groundtruth = int(np.argmax([self.first_stim,
                                          self.second_stim,
                                          self.third_stim]))
        self.reset()

    def reset(self):
        self.reward       = 0
        self.task_stage   = 'pre_delay'
        self.done         = False
        self.elapsed_t    = 0
        self.correct_trial = False

        self.first_stim  = self.rng.choice(self.stimulus_set)
        self.second_stim = self.select_second_stim()
        self.third_stim  = self.select_third_stim()
        self.pre_delay = self.rng.choice(self.pre_delay_set)

        self.groundtruth = int(np.argmax([self.first_stim,
                                          self.second_stim,
                                          self.third_stim]))

        self.observation = [0,0,0]
        self.rng.seed(np.random.randint(0, 2**32 - 1))
        return self.observation

    def step(self, action=None):
        obs, rew, done = None, 0, False

        if self.task_stage == "init":
            if action == 1:
                self.task_stage = "first_stim"
                self.observation = [1,0,0]
                self.reward = 1
            else:
                self.reward = -1
        elif self.task_stage == "pre_delay":
            if self.elapsed_t >= self.pre_delay:
                self.task_stage = "init"
                self.observation = [1,1,1]
                self.elapsed_t = 0
            else:
                self.elapsed_t += 1
        elif self.task_stage == "first_stim":
            if self.elapsed_t >= self.first_stim:
                self.task_stage = "delay_init"
                self.observation = [1,1,1]
                self.elapsed_t = 0
            else:
                self.elapsed_t += 1

        elif self.task_stage == "delay_init":
            self.task_stage = "delay"
            self.observation = [0,0,0]

        elif self.task_stage == "delay":
            if self.elapsed_t >= self.delay_duration:
                self.task_stage = "delay_end"
                self.observation = [1,1,1]
                self.elapsed_t = 0
            else:
                self.elapsed_t += 1

        elif self.task_stage == "delay_end":
            self.task_stage = "second_stim"
            self.observation = [0,1,0]

        elif self.task_stage == "second_stim":
            if self.elapsed_t >= self.second_stim:
                self.task_stage = "intermediate_choice_init"
                self.observation = [1, 1, 1]
                self.elapsed_t = 0
            else:
                self.elapsed_t += 1
        elif self.task_stage == "intermediate_choice_init":
            intermediate_groundtruth = int((self.first_stim < self.second_stim))

            if action is not None:
                if action == intermediate_groundtruth:
                    self.reward = self.rwd * 0.5
                    self.correct_intermediate_trial = True
                else:
                    self.reward = self.inc_rwd * 0.5
                    self.correct_intermediate_trial = False
                self.task_stage = "second_delay_init"
                self.observation = [0, 0, 0]
                self.elapsed_t = 0
            else:
                self.reward = self.inc_rwd * 0.1

        elif self.task_stage == "second_delay_init":
            self.task_stage = "second_delay"
            self.observation = [0,0,0]

        elif self.task_stage == "second_delay":
            if self.elapsed_t >= self.delay_duration:
                self.task_stage = "second_delay_end"
                self.observation = [1,1,1]
                self.elapsed_t = 0
            else:
                self.elapsed_t += 1

        elif self.task_stage == "second_delay_end":
            self.task_stage = "third_stim"
            self.observation = [0,0,1]

        elif self.task_stage == "third_stim":
            if self.elapsed_t >= self.third_stim:
                self.task_stage = "choice_init"
                self.observation = [1,1,1]
                self.elapsed_t = 0
            else:
                self.elapsed_t += 1

        elif self.task_stage == "choice_init":
            if action == self.groundtruth:
                self.reward = self.rwd
                self.correct_trial = True
            else:
                self.reward = self.inc_rwd
            self.done = True

        return self.observation, self.reward, self.done

    def calc_reward_without_stepping(self, action=None):
        if self.task_stage == "init":
            return 1 if action == 1 else -1
        elif self.task_stage == "choice_init":
            return self.rwd if action == self.groundtruth else self.inc_rwd
        else:
            return 0

    def select_second_stim(self):
        choices = copy.deepcopy(self.stimulus_set)
        choices.remove(self.first_stim)
        return self.rng.choice(choices)

    def select_third_stim(self):
        choices = copy.deepcopy(self.stimulus_set)
        choices.remove(self.first_stim)
        choices.remove(self.second_stim)
        return self.rng.choice(choices)

