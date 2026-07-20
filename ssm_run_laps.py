import random
import os
from re import I
from agents.model_1d import select_action, finish_trial
from agents.model_ssm_RL_laps import *
# from agents.model_ssm_RL_laps import finish_run, finish_readout
from envs.lap_counting import Laps_Counting
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import re
import sys
import wandb
from basic_lap_state import plot_lap_choices
from torch.distributions import Categorical
from train_and_plot_laps import collect_hidden_laps

sys.path.insert(1,'/nfs/turbo/coe-wluee/senlu/deepcell/')
from analysis import utils_linclab_plot
# utils_linclab_plot.linclab_plt_defaults(font="Arial", fontdir="analysis/fonts")

parser = argparse.ArgumentParser(description="Head-fixed Interval Discrimination task simulation")
parser.add_argument("--n_total_episodes",type=int,default=100000,help="Total episodes to train the model on task")
parser.add_argument("--save_ckpt_per_episodes",type=int,default=300,help="Save model every this number of episodes")
parser.add_argument("--record_data", default=False, action='store_true', help="Whether to collect data while training.")
parser.add_argument("--load_model_path", type=str, default='None'       , help="path RELATIVE TO $SCRATCH/timecell/training/timing")
parser.add_argument("--save_ckpts", action='store_true', default=False, help="Whether to save model every save_ckpt_per_epidoes episodes")
parser.add_argument("--n_neurons", type=int, default=50, help="Number of neurons in the LSTM layer and linear layer")
parser.add_argument("--lr", type=float, default=5e-3, help="learning rate")
parser.add_argument("--seed", type=int, default=2, help="seed to ensure reproducibility")
# parser.add_argument("--env_type", type=str, default='mem', help="type of environment: mem or nomem")
parser.add_argument("--hidden_type", type=str, default='ssm', help='type of hidden layer in the second last layer: lstm or linear')
parser.add_argument("--save_fig", default=False, action='store_true', help="If False, don't pass anything. If true, pass True.")
parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight_decay")
parser.add_argument("--entropy", type=float, default=0.1, help="entropy loss")
parser.add_argument("--p_dropout", type=float, default=0.0, help="dropout probability")
parser.add_argument("--dropout_type", type=int, default=None, help="location of dropout (could be 1,2,3,or 4)")
parser.add_argument("--wandb", default=False, action='store_true', help="Record on wandb.")
parser.add_argument("--spike", default=False, action='store_true', help="Spiking SSM.")
parser.add_argument("--lap_length", type=int, default=60, help="Running lap length.")
parser.add_argument("--lap_count", type=int, default=4, help="Running lap count.")
parser.add_argument("--server", default=False, action='store_true', help="Running on server.")
parser.add_argument("--layer2", default=False, action='store_true', help="Activate second layer.")
parser.add_argument("--eval", default=False, action='store_true', help="Evaluate only.")
parser.add_argument("--approx", action='store_true', default=False, help="whether to vary delay in training")
parser.add_argument("--readout", action='store_true', default=False, help="whether to train a readout")
parser.add_argument("--jitter", type=float, default=1.0, help="lap ends jitter")

args = parser.parse_args()
argsdict = args.__dict__
print(argsdict)

n_total_episodes = argsdict['n_total_episodes']
save_ckpt_per_episodes = argsdict['save_ckpt_per_episodes']
save_ckpts = True if argsdict['save_ckpts'] == True or argsdict['save_ckpts'] == 'True' else False
record_data = True if argsdict['record_data'] == True or argsdict['record_data'] == 'True' else False
load_model_path = argsdict['load_model_path']
window_size = n_total_episodes // 10
n_neurons = argsdict["n_neurons"]
# len_delay = argsdict['len_delay']
lr = argsdict['lr']
# env_type = argsdict['env_type']
hidden_type = argsdict['hidden_type']
seed = argsdict['seed']
save_performance_fig = True if argsdict['save_fig'] == True or argsdict['save_fig'] == 'True' else False
weight_decay = argsdict['weight_decay']
p_dropout = argsdict['p_dropout']
dropout_type = argsdict['dropout_type']
ent = argsdict['entropy']
wandb_log = argsdict['wandb']
spike = argsdict['spike']
lap_length = argsdict['lap_length']
lap_count = argsdict['lap_count']
server = argsdict['server']
layer2= argsdict['layer2']
eval_net = argsdict['eval']
approx = argsdict['approx']
readout = argsdict['readout']
eval = argsdict['eval']
jitter = argsdict['jitter']

if wandb_log:
    run = wandb.init(

        # Set the wandb project where this run will be logged.
        project="timecell",
        # Track hyperparameters and run metadata.
        config={
            "architecture": hidden_type,
            "dataset": "Running laps",
            "epochs": n_total_episodes,
            "state_dim": n_neurons,
            "spike": spike,
            "lap_length": lap_length,
            "lap_count": lap_count,
            "learning_rate": lr,
            "entropy": ent,
            "weight_decay": weight_decay,
            "layer2": layer2,
            "pretrained": load_model_path,
            "approx": approx,
            "jitter": jitter,
        },
    )
# Make directory in /training or /data_collecting to save data and model
if record_data:
    main_dir = './data_collecting/timing/laps'
    if not server:
        main_dir = '/home/lugroup/Documents/Sen_Code/deeprl-timecells/expts'+ main_dir[1:]
else:
    main_dir = './training/timing'
save_dir_str = f'{hidden_type}_{n_neurons}_{lr}'
if weight_decay != 0:
    save_dir_str += f'_wd{weight_decay}'
if p_dropout != 0:
    save_dir_str += f'_p{p_dropout}_{dropout_type}'
if spike:
    save_dir_str += f'_spiking'
if layer2:
    save_dir_str += f'_2layer'
save_dir_str += f'_lap{lap_length}'
save_dir = os.path.join(main_dir, save_dir_str)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
print(f'Saved to {save_dir}')

# Setting up cuda and seeds
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.manual_seed(seed)
np.random.seed(seed)

env = Laps_Counting(final_rwd=10,
                 seed=seed,
                 lap_length=lap_length,
                 fixed_laps=lap_count,
                 approx=approx,
                 jitter=jitter,
                 randomize_laps=True)

if hidden_type =="ssm":
    ssm_params = {
            "P": n_neurons*2,
            "C_init": "trunc_standard_normal",
            "discretization": "zoh",
            "dt_min": 0.001,
            "dt_max": 0.1,
            "conj_sym": True,
            "step_rescale": 1.0,
            "spike":spike,
            "layer2":layer2,
            "lap_count":lap_count,
        }
    # Instantiate the AC network using SSM as core.
    net = AC_SSM_stack(input_dimensions=2, action_dimensions=2, batch_size=1,
                 hidden_dim=n_neurons, ssm_params=ssm_params, p_dropout=0.1).to(device)
else:
    print(f"{hidden_type} not implemented.")
    exit(0)
# Reset hidden state at episode start.
net.reinit_hid()

env_title = "Running Laps"
net_title = 'S5'
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

# Load existing model
if load_model_path=='None':
    ckpt_name = f'seed_{seed}_agent'  # placeholder ckptname in case we want to save data in the end
else:
    print(f"Loading model from {load_model_path}")
    net.load_state_dict(torch.load(load_model_path))

def get_param_linear(epoch, start=1.0, end=0.1, total_epochs=5000):
    """
    Linearly interpolate p from `start`→`end` over epochs 1→total_epochs.
    """
    if epoch <= 1:
        return start
    elif epoch >= total_epochs:
        return end
    # fraction of the way through (0.0 at epoch=1, 1.0 at epoch=total_epochs)
    frac = (epoch - 1) / (total_epochs - 1)
    return start + (end - start) * frac

# Define helper functions
def bin_rewards(epi_rewards, window_size):
    """
    Average the epi_rewards with a moving window.
    """
    epi_rewards = epi_rewards.astype(np.float32)
    avg_rewards = np.zeros_like(epi_rewards)
    for i_episode in range(1, len(epi_rewards)+1):
        if 1 < i_episode < window_size:
            avg_rewards[i_episode-1] = np.mean(epi_rewards[:i_episode])
        elif window_size <= i_episode <= len(epi_rewards):
            avg_rewards[i_episode-1] = np.mean(epi_rewards[i_episode - window_size: i_episode])
    return avg_rewards

def plot_neuron_outputs(readout_logits: np.ndarray,
                        episode_idx: int = 0,
                        average_episodes: bool = False,
                        save_path: str | None = None):
    """
    Plot the four-neuron read-out logits over time.

    Parameters
    ----------
    readout_logits : np.ndarray
        Shape (n_eps, time_steps, 4).
    episode_idx : int, default 0
        Which episode to plot if `average_episodes` is False.
    average_episodes : bool, default False
        If True, plots the mean curve across episodes instead.
    save_path : str | None, default None
        If given, saves the figure to this path instead of `plt.show()`.
    """
    if average_episodes:
        data = readout_logits.mean(axis=0)          # (time_steps, 4)
        title = f"Average across {readout_logits.shape[0]} episodes"
    else:
        data = readout_logits[episode_idx]          # (time_steps, 4)
        title = f"Episode {episode_idx}"

    t = np.arange(data.shape[0])

    plt.figure(figsize=(10, 6))
    for k in range(4):
        plt.plot(t, data[:, k], label=f'Neuron {k+1}')
    plt.xlabel('Time step')
    plt.ylabel('Read-out logit')
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path+f'readout_logits.png', dpi=200)
        print(f"Figure saved to {save_path}")
        plt.close()
    else:
        plt.show()
def vp_distance(pred, gt, q):
    """
    Victor–Purpura distance between two 1‑D event lists.
    pred, gt: 1‑D numpy arrays of event times (in seconds, sorted).
    q: cost per second for shifting an event.
    """
    m, n = len(pred), len(gt)
    # DP matrix: (m+1)×(n+1)
    D = np.zeros((m + 1, n + 1))
    D[0, :] = np.arange(n + 1)            # cost of n inserts
    D[:, 0] = np.arange(m + 1)            # cost of m deletes

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost_shift = D[i - 1, j - 1] + q * abs(pred[i - 1] - gt[j - 1])
            cost_del   = D[i - 1, j] + 1
            cost_ins   = D[i, j - 1] + 1
            D[i, j] = min(cost_shift, cost_del, cost_ins)

    return D[m, n]

def vp_score(pred, gt, q):
    dist = vp_distance(np.asarray(pred), np.asarray(gt), q)
    return 1.0 - dist / (len(pred) + len(gt))

def train_readout(network,lr, env, n_eps=500, lap_count=4, savepath=None):
    # network.add_probe()
    optimizer = torch.optim.Adam(network.linear_probe.parameters(), lr=lr)
    corr,total = 0, 0
    for i_episode in tqdm(range(n_eps)):
        done = False
        observation = env.reset()
        if hasattr(network, 'reinit_hid'):
            network.reinit_hid()
        while not done:
            # The observation needs to be a tensor for the network
            obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(device)
            probe_output = network.forward_linear_probe(obs_tensor)
            label, new_obs, task_stage = env.step_probe_no_reward()
            done = (task_stage == 'done')  # The episode is 'done' only when the task_stage says so.
            observation = new_obs
            pred, loss =finish_readout(network, probe_output, label, optimizer, max(label)==1)
            # print(f'Loss: {loss:0.3f}')
            if max(label)==1 :
                total+=1
                if pred == np.argmax(label):
                    corr+=1
        print(f'acc = {corr/total:.2f}, loss: {loss:.2f}')
    with torch.no_grad():
        readout_logits = np.zeros([n_eps, lap_length*lap_count+lap_count, lap_count], dtype=np.float64)
        for i_episode in tqdm(range(n_eps)):
            done = False
            observation = env.reset()
            if hasattr(network, 'reinit_hid'):
                network.reinit_hid()
            while not done:
                # The observation needs to be a tensor for the network
                obs_tensor = torch.tensor([observation], dtype=torch.float32).unsqueeze(0).to(device)
                probe_output = network.forward_linear_probe(obs_tensor)
                # print(obs_tensor)
                # print(probe_output)
                readout_logits[i_episode, env.elapsed_t] = probe_output.cpu().numpy()
                label, new_obs, task_stage = env.step_probe_no_reward()
                done = (task_stage == 'done')  # The episode is 'done' only when the task_stage says so.
                observation = new_obs
    plot_neuron_outputs(readout_logits, average_episodes=True, save_path=savepath)
    exit(0)


if readout:
    # pretrained_lap='/home/lugroup/Documents/Sen_Code/deeprl-timecells/expts/data_collecting/timing/laps/seed_2_run_lap_91.000.pt'
    net.load_state_dict(torch.load(load_model_path))
    train_readout(net, 1.5, env,n_eps=10000,lap_count=lap_count, savepath=save_dir)
    exit(0)

intermediate_action_hist = np.zeros(n_total_episodes, dtype=np.int8)
correct_intermediate_trial_hist = np.zeros(n_total_episodes, dtype=np.int8)
episode_saved_actions = []
episode_rewards = []

action_hist = np.zeros(n_total_episodes, dtype=np.int8)
correct_trial = np.zeros(n_total_episodes, dtype=np.int8)
stim = np.zeros((n_total_episodes, 3), dtype=np.int8)
policy_hist, loss_hist, p_loss_hist, v_loss_hist = [], [], [], []
vps = []
highest=-1.0
if eval:
    net.eval()
else:
    for i_episode in tqdm(range(n_total_episodes)):
        done = False
        observation = env.reset()
        if hasattr(net, 'reinit_hid'):
            net.reinit_hid()

        # Lists to store data for the current episode
        episode_rewards = []
        episode_saved_actions = []
        intermediate_corrects = [] # Tracks correct (1) vs incorrect (0) lap reports within the episode
        pred = []
        while not done:
            # The observation needs to be a tensor for the network
            obs_tensor = torch.tensor([observation], dtype=torch.float32).to(device)

            # Forward pass through the network
            if spike:
                # The original forward pass calls are kept, assuming they match your network's interface
                if layer2:
                    pol, val, lin_act, _,_ = net.forward(obs_tensor)
                else:
                    pol, val, lin_act, _ = net.forward(obs_tensor)

            else:
                pol, val, lin_act = net.forward(obs_tensor)

            # Select action based on the policy
            # MODIFICATION: Assumed `select_action` returns both the integer action and a
            # `SavedAction` object (e.g., a tuple with log_prob and value) needed for `finish_trial`.
            # Create a categorical distribution over the policy (action probabilities)
            dist = Categorical(pol)

            # Sample an action from the distribution
            action_tensor = dist.sample()
            action_to_take = action_tensor.item()
            # print(action_to_take)
            # Save the log probability of the action and the state value for later use in training
            log_prob = dist.log_prob(action_tensor)
            episode_saved_actions.append(SavedAction(log_prob, val, pol))

            if action_to_take==1:
                pred.append(env.elapsed_t)
            # Take action in the environment
            # MODIFICATION: The `step` function from `lap_counting.py` returns obs, reward, and task_stage.
            new_obs, reward, task_stage = env.step2(action_to_take)
            # print(reward)
            done = (task_stage == 'done') # The episode is 'done' only when the task_stage says so.
            episode_rewards.append(reward)

            # The reward from the environment directly tells us if an intermediate report was correct.
            # A reward of 1.0 means a lap was correctly reported.
            # A reward of -1.0 means it was incorrectly reported (or missed).




            # Update the environment observation for the next timestep
            observation = new_obs

        vps.append(vp_score(pred, env.lap_ends, 1))
        # print(f'vp score = {vps[-1]}')
        # --- End of Episode ---

        # Update trial-level correctness metrics after the episode is complete
        correct_trial[i_episode] = 1 if env.predicted_lap_count == env.true_lap_count else 0
        # print(f'predicted lap count {env.predicted_lap_count} total reward {sum(episode_rewards)}')
        # Prepare network for learning by assigning the collected episode data
        net.rewards = episode_rewards
        net.saved_actions = episode_saved_actions

        if eval: # If in evaluation mode, don't learn, just clear the data
            if hasattr(net, 'rewards'): del net.rewards[:]
            if hasattr(net, 'saved_actions'): del net.saved_actions[:]
        else:
            # Perform backpropagation and update network weights
            p_loss, v_loss, total_loss = finish_run(net, 1, optimizer, entropy_weight=get_param_linear(i_episode + 1,start=ent, total_epochs=n_total_episodes))
            loss_hist.append(total_loss)
            p_loss_hist.append(p_loss)
            v_loss_hist.append(v_loss)

        if (i_episode + 1) % 500 == 0:
            print(f'Highest = {highest:.3f} %')
        # Logging block
        if (i_episode + 1) % 100 == 0:
            # Ensure we don't index out of bounds on the first 100 episodes
            # last_100_intermediate = correct_intermediate_trial_hist[max(0, len(correct_intermediate_trial_hist) - 100):]
            # last_1000_final = correct_trial[max(0, len(correct_trial) - 1000):]
            last_100_final = correct_trial[max(0, i_episode - 100):i_episode]

            # avg_intermediate_correct = np.mean(last_100_intermediate) * 100
            # avg_1000_correct = np.mean(last_1000_final) * 100
            avg_final_correct =  np.mean(correct_trial[:i_episode]) * 100
            avg_100_correct = np.mean(last_100_final) * 100

            print(
                f'Episode {i_episode + 1}, '
                # f'{avg_intermediate_correct:.3f}% intermediate correct (last 100), '
                # f'{avg_final_correct:.3f}% final correct (total average), '
                f'vp score: {sum(vps[-100:])/len(vps[-100:]):.3f}, '
                f'{avg_100_correct:.3f}% final correct (last 100)'
            )
            if avg_100_correct>highest:
                highest = avg_100_correct
                torch.save(net.state_dict(),
                           save_dir + f'/seed_{argsdict["seed"]}_run_{lap_count}_lap_best.pt')

            if wandb_log and not eval_net:
                wandb.log({
                    "episode": i_episode + 1,
                    "intermediate_acc": avg_intermediate_correct,
                    "acc": avg_final_correct,
                    "total_loss": total_loss.item(),
                    "policy_loss": p_loss.item(),
                    "value_loss": v_loss.item(),
                })
        if (i_episode+1) % 100 == 0 and not record_data:
            avg_intermediate_correct = np.mean(correct_intermediate_trial_hist[max(0, i_episode - 99):i_episode + 1]) * 100
            avg_final_correct = np.mean(correct_trial[max(0, i_episode - 99):i_episode + 1]) * 100
            if not server:
                # plot_loss_and_policy2(
                #     loss_list=[np.array(p_loss_hist), np.array(v_loss_hist), np.array(loss_hist)],
                #     policy_hist=np.array(policy_hist),
                #     save_path="/home/lugroup/Documents/Sen_Code/deeprl-timecells/ssm_3stim_loss_latest.png",
                #     loss_labels=["p_loss", "v_loss", "total loss"],
                #     policy_labels=["0", "1", "2"],
                #     smooth_window=100
                # )
                print(f'accuracy {avg_final_correct}\%')
            if wandb_log:
                wandb.log({
                    "episode": i_episode + 1,
                    "intermediate_acc": avg_intermediate_correct,
                    "acc": avg_final_correct,
                    "total_loss": total_loss.item(),
                    "policy_loss": p_loss.item(),
                    "value_loss": v_loss.item(),
                    # Log policy probabilities if needed, e.g., average policy over the last 100 episodes
                    # "avg_policy_last_100": np.mean(np.array(policy_hist)[max(0, len(policy_hist)-100):], axis=0)
                })
        if (i_episode+1) % save_ckpt_per_episodes == 0:
            if load_model_path != 'None':
                print(f'Episode {i_episode+49999}, '
                      f'{np.mean(correct_trial[i_episode+1-save_ckpt_per_episodes:i_episode+1])*100:.3f}% '
                      f'correct in the last {save_ckpt_per_episodes} episodes, '
                      f'avg {np.mean(correct_trial[:i_episode+1])*100:.3f}% correct')
            else:
                print(f'Episode {i_episode}, {np.mean(correct_trial[i_episode+1-save_ckpt_per_episodes:i_episode+1])*100:.3f}% correct in the last {save_ckpt_per_episodes} episodes, avg {np.mean(correct_trial[:i_episode+1])*100:.3f}% correct')
            if save_ckpts:
                if load_model_path != 'None':
                    torch.save(net.state_dict(), save_dir + f'/seed_{argsdict["seed"]}_epi{i_episode+49999}.pt')
                else:
                    torch.save(net.state_dict(), save_dir + f'/seed_{argsdict["seed"]}_epi{i_episode}.pt')
# torch.save(net.state_dict(), save_dir + f'/seed_{argsdict["seed"]}_run_{lap_count}_lap_{avg_100_correct:.3f}.pt')
binned_correct_trial = bin_rewards(correct_trial, window_size)

fig, ax = plt.subplots()
fig.suptitle(env_title)
ax.plot(np.arange(n_total_episodes), binned_correct_trial, label=net_title)
ax.set_xlabel("Episode")
ax.set_ylabel("Correct rate")
ax.set_ylim(0,1)
ax.legend(frameon=False)

plot_lap_choices(net, env, save_path=save_dir)
if eval:
    exit(0)
if record_data:
    action_hist = np.zeros(n_total_episodes, dtype=np.int8)
    correct_trial = np.zeros(n_total_episodes, dtype=np.int8)
    policy_hist, loss_hist, p_loss_hist, v_loss_hist = [], [], [], []
    eval_eps = n_total_episodes // 10

    # Collect hidden states and spiking data using the new function
    full_resp1, full_resp2, spiking_entries1, spiking_entries2 = collect_hidden_laps(
        net=net,
        env=env,
        device=device,
        n_episodes=eval_eps,
        lap_length=lap_length,
        lap_count=lap_count,
        spike=spike,
        layer2=layer2,
    )


if record_data:
    if layer2:
        np.savez_compressed(save_dir + f'/{ckpt_name}_hidden2_state.npz', action_hist=action_hist, correct_trial=correct_trial,hidden1=full_resp1, hidden2=full_resp2)
    else:
        np.savez_compressed(save_dir + f'/{ckpt_name}_hidden1_state.npz', action_hist=action_hist, correct_trial=correct_trial,hidden1=full_resp1)
else: 
    np.savez_compressed(save_dir + f'/seed_{seed}_total_{n_total_episodes}episodes_performance_data.npz', action_hist=action_hist, correct_trial=correct_trial, stim=stim)
