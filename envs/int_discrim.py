import numpy as np
from numpy import array
import random
import copy
# from gymnasium import spaces
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
        self.action_space = spaces.Discrete(2)      # Boolean variable that stim_1 > stim_2
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
        self.groundtruth = self.first_stim > self.second_stim  # 1 if L1>L2, 0 if L1<L2
        self.elapsed_t = 0
        self.correct_trial = False
        self.observation = [1,1]

    def step(self, action=None):
        """
        :param action
        :return: observation, reward, done, info
        """

        if self.task_stage == "init":               # the agent needs to take an action
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

        elif self.task_stage == "choice_init":                      # the agent needs to take an action
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

        if self.task_stage == "init":               # the agent needs to take an action
            if action == 1:
                reward = 1
            else:
                reward = -1
        elif self.task_stage == "choice_init":                      # the agent needs to take an action
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
        # self.stimulus_set   = [1,1,1]
        self.delay_duration = 50

        # now three choices
        self.action_space      = spaces.Discrete(3)
        self.observation_space = spaces.MultiBinary(3)

        self.rng     = np.random.RandomState(seed)
        self.rwd     = rwd
        self.inc_rwd = inc_rwd

        # pick all three stimuli
        self.first_stim  = self.rng.choice(self.stimulus_set)
        self.second_stim = self.select_second_stim()
        self.third_stim  = self.select_third_stim()

        # index of the longest interval
        self.groundtruth = int(np.argmax([self.first_stim,
                                          self.second_stim,
                                          self.third_stim]))

        # initialize dynamics
        self.reset()

    def reset(self):
        self.reward       = 0
        self.task_stage   = 'init'
        self.done         = False
        self.elapsed_t    = 0
        self.correct_trial = False

        # draw three distinct stimuli
        self.first_stim  = self.rng.choice(self.stimulus_set)
        self.second_stim = self.select_second_stim()
        self.third_stim  = self.select_third_stim()

        self.groundtruth = int(np.argmax([self.first_stim,
                                          self.second_stim,
                                          self.third_stim]))

        # all three “cue” lights on before start
        self.observation = [1,1,1]
        return self.observation

    def step(self, action=None):
        obs, rew, done = None, 0, False

        if self.task_stage == "init":
            # poke to start
            if action == 1:
                self.task_stage = "first_stim"
                self.observation = [1,0,0]
                self.reward = 1
            else:
                self.reward = -1

        elif self.task_stage == "first_stim":
            # self.reward = 0
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
            # self.reward = 1

        elif self.task_stage == "second_stim":
            # self.reward = 0
            if self.elapsed_t >= self.second_stim:
                # begin second delay before third stim
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
            # self.reward=1

        elif self.task_stage == "third_stim":
            # self.reward = 0
            if self.elapsed_t >= self.third_stim:
                self.task_stage = "choice_init"
                self.observation = [1,1,1]
                self.elapsed_t = 0
            else:
                self.elapsed_t += 1

        elif self.task_stage == "choice_init":
            # make the choice among three
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
        # self.stimulus_set   = [1,1,1]
        if fixed_delay:
            self.delay_set = [delay]
        else:
            self.delay_set = list(range(10, delay, 10))
        # now three choices
        self.action_space      = spaces.Discrete(3)
        self.observation_space = spaces.MultiBinary(3)

        self.rng     = np.random.RandomState(seed)
        self.rwd     = rwd
        self.inc_rwd = inc_rwd


        # pick all three stimuli
        delay_c = self.rng.choice(self.delay_set)
        self.delay_duration = int(delay_c * multiplier)
        self.first_stim  = self.rng.choice(self.stimulus_set)
        self.second_stim = self.select_second_stim()
        self.third_stim  = self.select_third_stim()

        # index of the longest interval
        self.groundtruth = int(np.argmax([self.first_stim,
                                          self.second_stim,
                                          self.third_stim]))
        # initialize dynamics
        self.reset()

    def reset(self):
        self.rng.seed(np.random.randint(0, 2**32 - 1))
        self.reward       = 0
        self.task_stage   = 'init'
        self.done         = False
        self.elapsed_t    = 0
        self.correct_trial = False

        # draw three distinct stimuli
        self.first_stim  = self.rng.choice(self.stimulus_set)
        self.second_stim = self.select_second_stim()
        self.third_stim  = self.select_third_stim()


        self.groundtruth = int(np.argmax([self.first_stim,
                                          self.second_stim,
                                          self.third_stim]))
        self.delay_duration1 = self.rng.choice(self.delay_set)
        self.delay_duration2 = self.rng.choice(self.delay_set)
        # all three “cue” lights on before start
        self.observation = [1,1,1]
        return self.observation

    def step(self, action=None):
        obs, rew, done = None, 0, False

        if self.task_stage == "init":
            # poke to start
            if action == 1:
                self.task_stage = "first_stim"
                self.observation = [1,0,0]
                self.reward = 1
            else:
                self.reward = -1

        elif self.task_stage == "first_stim":
            # self.reward = 0
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
            # self.reward = 1


        elif self.task_stage == "second_stim":
            # self.reward = 0 # Keep reward at 0 during stimulus presentation
            if self.elapsed_t >= self.second_stim:
                self.task_stage = "intermediate_choice_init"  # New stage
                self.observation = [1, 1, 1]  # Provide a cue for intermediate choice
                self.elapsed_t = 0
            else:
                self.elapsed_t += 1
        elif self.task_stage == "intermediate_choice_init":  # New intermediate choice stage
            # Agent makes a choice comparing stim 1 and stim 2
            # Action 0 could mean stim1 <= stim2, Action 1 could mean stim1 > stim2
            intermediate_groundtruth = int((self.first_stim < self.second_stim)) # 1 if L1 > L2, 0 if L1 <= L2

            if action is not None:  # Ensure an action was provided
                if action == intermediate_groundtruth:
                    self.reward = self.rwd * 0.5  # Assign a partial reward (adjust weight as needed)
                    self.correct_intermediate_trial = True  # Track intermediate correctness
                else:
                    self.reward = self.inc_rwd * 0.5  # Assign a partial penalty (adjust weight as needed)
                    self.correct_intermediate_trial = False
                self.task_stage = "second_delay_init"  # Move to the second delay
                self.observation = [0, 0, 0]  # Transition observation
                self.elapsed_t = 0
            else:
                # Handle case where no action is provided - maybe a penalty or just stay in stage
                # For simplicity, we assume an action is always provided by the agent in the training loop
                self.reward = self.inc_rwd * 0.1  # Small penalty for no action

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
            # self.reward=1

        elif self.task_stage == "third_stim":
            # self.reward = 0
            if self.elapsed_t >= self.third_stim:
                self.task_stage = "choice_init"
                self.observation = [1,1,1]
                self.elapsed_t = 0
            else:
                self.elapsed_t += 1

        elif self.task_stage == "choice_init":
            # make the choice among three
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
        # self.stimulus_set   = [1,1,1]
        self.delay_duration = delay

        # now three choices
        self.action_space      = spaces.Discrete(3)
        self.observation_space = spaces.MultiBinary(3)

        self.rng     = np.random.RandomState(seed)
        self.rwd     = rwd
        self.inc_rwd = inc_rwd


        # pick all three stimuli
        self.pre_delay = self.rng.choice(self.pre_delay_set)
        self.first_stim  = self.rng.choice(self.stimulus_set)
        self.second_stim = self.select_second_stim()
        self.third_stim  = self.select_third_stim()

        # index of the longest interval
        self.groundtruth = int(np.argmax([self.first_stim,
                                          self.second_stim,
                                          self.third_stim]))
        # initialize dynamics
        self.reset()

    def reset(self):
        self.reward       = 0
        self.task_stage   = 'pre_delay'
        self.done         = False
        self.elapsed_t    = 0
        self.correct_trial = False

        # draw three distinct stimuli
        self.first_stim  = self.rng.choice(self.stimulus_set)
        self.second_stim = self.select_second_stim()
        self.third_stim  = self.select_third_stim()
        self.pre_delay = self.rng.choice(self.pre_delay_set)

        self.groundtruth = int(np.argmax([self.first_stim,
                                          self.second_stim,
                                          self.third_stim]))

        # all three “cue” lights on before start
        self.observation = [0,0,0]
        self.rng.seed(np.random.randint(0, 2**32 - 1))
        return self.observation

    def step(self, action=None):
        obs, rew, done = None, 0, False

        if self.task_stage == "init":
            # poke to start
            if action == 1:
                self.task_stage = "first_stim"
                self.observation = [1,0,0]
                self.reward = 1
            else:
                self.reward = -1
        elif self.task_stage == "pre_delay":
            # self.reward = 0
            if self.elapsed_t >= self.pre_delay:
                self.task_stage = "init"
                self.observation = [1,1,1]
                self.elapsed_t = 0
            else:
                self.elapsed_t += 1
        elif self.task_stage == "first_stim":
            # self.reward = 0
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
            # self.reward = 1


        elif self.task_stage == "second_stim":
            # self.reward = 0 # Keep reward at 0 during stimulus presentation
            if self.elapsed_t >= self.second_stim:
                self.task_stage = "intermediate_choice_init"  # New stage
                self.observation = [1, 1, 1]  # Provide a cue for intermediate choice
                self.elapsed_t = 0
            else:
                self.elapsed_t += 1
        elif self.task_stage == "intermediate_choice_init":  # New intermediate choice stage
            # Agent makes a choice comparing stim 1 and stim 2
            # Action 0 could mean stim1 <= stim2, Action 1 could mean stim1 > stim2
            intermediate_groundtruth = int((self.first_stim < self.second_stim)) # 1 if L1 > L2, 0 if L1 <= L2

            if action is not None:  # Ensure an action was provided
                if action == intermediate_groundtruth:
                    self.reward = self.rwd * 0.5  # Assign a partial reward (adjust weight as needed)
                    self.correct_intermediate_trial = True  # Track intermediate correctness
                else:
                    self.reward = self.inc_rwd * 0.5  # Assign a partial penalty (adjust weight as needed)
                    self.correct_intermediate_trial = False
                self.task_stage = "second_delay_init"  # Move to the second delay
                self.observation = [0, 0, 0]  # Transition observation
                self.elapsed_t = 0
            else:
                # Handle case where no action is provided - maybe a penalty or just stay in stage
                # For simplicity, we assume an action is always provided by the agent in the training loop
                self.reward = self.inc_rwd * 0.1  # Small penalty for no action

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
            # self.reward=1

        elif self.task_stage == "third_stim":
            # self.reward = 0
            if self.elapsed_t >= self.third_stim:
                self.task_stage = "choice_init"
                self.observation = [1,1,1]
                self.elapsed_t = 0
            else:
                self.elapsed_t += 1

        elif self.task_stage == "choice_init":
            # make the choice among three
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
