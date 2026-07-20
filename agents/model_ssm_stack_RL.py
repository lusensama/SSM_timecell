"""
model_s5ssm.py

This module implements an SSM-based actor–critic network with HIPPO-based initialization.
It contains:
 - Discretization and HIPPO initialization utilities.
 - S5SSMCell: a single-step state-space model cell.
 - AC_SSM: an actor–critic network that uses S5SSMCell as its recurrent core,
           modified to stack two SSM cells.

Differences vs `agents/model_ssm_RL_laps.py` (lap-specific variant):
- Pure PyTorch: does not import or depend on JAX; uses `agents.ssm_init` directly.
- Spiking mode: uses a simple surrogate gradient spike (SpikeSTE) without LIF membrane state.
- Device handling: avoids hardcoded devices; internal tensors follow `self.device`.
- Training helper: provides `finish_trial` (no return normalization); lacks `finish_run` alias.
- No probe utilities: does not include `add_probe`, `forward_linear_probe`, or `finish_readout`.
- Forward API: returns `(policy, value, lin_act)`; if spiking, may return extra intermediates for logging.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from collections import namedtuple
from agents import ssm_init
##########################################
# HIPPO Initialization & Discretization Utilities
##########################################
# Named tuple to store log probability and state value for each action.
SavedAction = namedtuple('SavedAction', ['log_prob', 'value', 'policy'])

def discount_rwds(r, gamma):
    """
    Compute discounted rewards.

    Args:
        r (np.array): an array of rewards (one per time step).
        gamma (float): discount factor (e.g., 0.99).

    Returns:
        np.array: array of discounted rewards.
    """
    # Create an array of the same shape as r (converted to float).
    disc_rwds = np.zeros_like(r, dtype=float)
    running_add = 0.0
    for t in reversed(range(len(r))):
        running_add = running_add * gamma + r[t] # Corrected calculation
        disc_rwds[t] = running_add
    return disc_rwds

def select_action(model, policy_, value_):
    """
    Samples an action from the given policy, saves the log probability and state value,
    and returns the sampled action and additional info.

    Args:
        model: the actor–critic model which should contain an attribute 'saved_actions'.
        policy_ (torch.Tensor): Tensor containing action probabilities.
        value_ (torch.Tensor): Tensor containing the state value estimate.

    Returns:
        action (int): the sampled action (as a python integer).
        policy_info: additional policy info (e.g., first element of the tensor for logging).
        value_info: the state value (as a python float).
    """
    a = Categorical(policy_)
    action = a.sample()
    model.saved_actions.append(SavedAction(a.log_prob(action), value_, policy_))
    return action.item(), policy_.data[0], value_.squeeze().item()

def finish_trial(model, discount_factor, optimizer, scheduler=None,entropy_weight=0.5, **kwargs):
    """
    Computes the discounted rewards and loss for all saved actions in the trial,
    performs backpropagation, and updates the model's parameters.

    Args:
        model: the actor–critic model; it should have attributes:
                - rewards: list of collected rewards during the trial.
                - saved_actions: list of SavedAction namedtuples.
                - device: torch.device in which the model resides.
        discount_factor (float): discount factor (gamma) to use.
        optimizer: the optimizer for updating the model parameters.
        scheduler (optional): learning rate scheduler.

    Returns:
        Tuple of (policy_loss, value_loss) as torch Tensors.
    """
    # Convert rewards to a numpy array and compute discounted rewards.
    returns_ = discount_rwds(np.asarray(model.rewards), gamma=discount_factor)
    saved_actions = model.saved_actions

    policy_losses = []
    value_losses = []
    entropy_terms = []  # Add a list to store entropy terms

    # Convert discounted rewards to a tensor on the appropriate device.
    returns_ = torch.Tensor(returns_).to(model.device)

    for (log_prob, value, policy_), r in zip(saved_actions, returns_):
        # Compute reward prediction error
        rpe = r - value.item() # Corrected RPE calculation
        policy_losses.append(-log_prob * rpe)
        # Compute value loss using smooth L1 loss.
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([[r]], device=model.device))
                             .unsqueeze(-1))
        entropy = -(policy_ * torch.log(policy_ + 1e-8)).sum() # Add a small epsilon for numerical stability
        entropy_terms.append(entropy)
    optimizer.zero_grad()
    p_loss = torch.cat(policy_losses).sum()
    v_loss = torch.cat(value_losses).sum()
    if entropy_terms:
        entropy_bonus = -entropy_weight * torch.stack(entropy_terms).mean()
    else:
        entropy_bonus = 0
    total_loss = p_loss + v_loss + entropy_bonus

    total_loss.backward(retain_graph=True) # Removed retain_graph=True as it's usually not needed after a trial
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    # Clear the lists for the next trial.
    del model.rewards[:]
    del model.saved_actions[:]

    return p_loss, v_loss, total_loss

def discretize_zoh(Lambda, step_delta, time_delta):
    """
    Discretize a diagonal (complex) state matrix using the zero-order hold method.
    Args:
        Lambda: torch.complex tensor of shape (P,)
        step_delta: torch tensor of shape (P,) (real)
        time_delta: scalar float (integration timestep)
    Returns:
        Lambda_bar, gamma_bar – both torch.complex tensors of shape (P,)
    """
    Identity = torch.ones(Lambda.shape, dtype=Lambda.dtype, device=Lambda.device)
    Delta = step_delta * time_delta
    Lambda_bar = torch.exp(Lambda * Delta)
    gamma_bar = (1.0 / Lambda) * (Lambda_bar - Identity)
    return Lambda_bar, gamma_bar


def discretize_dirac(Lambda, step_delta, time_delta):
    """
    Discretize with Dirac-delta input spikes.
    """
    Delta = step_delta * time_delta
    Lambda_bar = torch.exp(Lambda * Delta)
    gamma_bar = torch.tensor(1.0, dtype=Lambda.dtype, device=Lambda.device)
    return Lambda_bar, gamma_bar


def discretize_async(Lambda, step_delta, time_delta):
    """
    Discretize with asynchronous (Dirac-delta with normalization) method.
    """
    Identity = torch.ones(Lambda.shape, dtype=Lambda.dtype, device=Lambda.device)
    Lambda_bar = torch.exp(Lambda * step_delta * time_delta)
    gamma_bar = (1.0 / Lambda) * (torch.exp(Lambda * step_delta) - Identity)
    return Lambda_bar, gamma_bar


def make_HiPPO(N):
    """
    Create a HiPPO-LegS matrix.
    Args:
        N: int, state size.
    Returns:
        A: (N, N) HiPPO-LegS matrix (torch.float32)
    """
    n = torch.arange(N, dtype=torch.float32)
    P_vec = torch.sqrt(1 + 2 * n)
    A = torch.tril(P_vec[:, None] * P_vec[None, :]) - torch.diag(n)
    return -A


def make_NPLR_HiPPO(N):
    """
    Compute components needed for the NPLR representation of HiPPO-LegS.
    Returns:
        A: HiPPO matrix (N x N)
        P_vec: (N,) low-rank vector
        B: (N,) HiPPO input vector
    """
    n = torch.arange(N, dtype=torch.float32)
    A = make_HiPPO(N)
    P_vec = torch.sqrt(n + 0.5)
    B = torch.sqrt(2 * n + 1.0)
    return A, P_vec, B


def make_DPLR_HiPPO(N):
    """
    Compute components for a DPLR representation of HiPPO-LegS.
    We return the diagonal (eigenvalue) part along with additional factors.
    Returns:
        Lambda: (N,) complex eigenvalues.
        P_transformed: transformed low-rank term.
        B_transformed: (N,) complex, input projection vector.
        V: eigenvector matrix.
        B_orig: original B vector (complex).
    """
    A, P_vec, B = make_NPLR_HiPPO(N)
    # S = A + P_vec outer-product P_vec
    S = A + torch.outer(P_vec, P_vec)
    S_diag = torch.diag(S)
    Lambda_real = S_diag.mean() * torch.ones_like(S_diag)

    # Compute eigen-decomposition of S * (-1j), yielding complex eigenvalues.
    S_complex = S.to(torch.complex64) * (-1j)
    eigvals, V = torch.linalg.eig(S_complex)
    Lambda_imag = eigvals.imag
    V_conj = V.conj().t()
    P_transformed = V_conj @ P_vec.to(torch.complex64)
    B_orig = B.to(torch.complex64)
    B_transformed = V_conj @ B_orig

    Lambda = Lambda_real.to(torch.complex64) + 1j * Lambda_imag
    return Lambda, P_transformed, B_transformed, V, B_orig

class SpikeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha= ctx.saved_tensors
        # piecewise-linear surrogate: nonzero gradient in |x|<1
        # surrogate_grad = torch.clamp(1.0 - x.abs(), min=0.0)
        sigma = torch.sigmoid(alpha * x)
        surrogate_grad = alpha * sigma * (1 - sigma)
        return grad_output * surrogate_grad, (grad_output * sigma *(1-sigma) ).sum()

##########################################
# SSM Cell and Actor-Critic Network
##########################################

class S5SSMCell(nn.Module):
    """
    A single-step state-space model cell using HIPPO initialization.
    This cell implements:
        new_state = Lambda_bar * state + gamma_bar * (B @ u)
        output    = Re( C_tilde @ new_state ) [+ feedthrough]
    The discretized matrices are computed on the fly.

    Args:
        H_in (int): input dimension.
        H_out (int): output dimension.
        P (int): full state dimension.
        C_init (str): initialization method for C ("trunc_standard_normal" or "lecun_normal").
        discretization (str): "zoh", "dirac", or "async".
        dt_min, dt_max (float): range for the learnable time constants.
        conj_sym (bool): if True, uses half the state (enforcing conjugate symmetry).
        step_rescale (float): multiplier for learned time constants.
    """
    def __init__(self, H_in, H_out, P,
                 C_init="trunc_standard_normal",
                 discretization="zoh",
                 dt_min=0.001, dt_max=0.1,
                 conj_sym=True, step_rescale=1.0):
        super().__init__()
        self.H_in = H_in
        self.H_out = H_out
        self.P = P
        self.conj_sym = conj_sym
        self.step_rescale = step_rescale
        self.lambda_bar = None

        # Initialize HIPPO components for a single block of size P.
        # Lambda, _, B, _, _ = make_DPLR_HiPPO(P)
        Lambda, _, B, _, _ = ssm_init.make_DPLR_HiPPO(P)
        if self.conj_sym:
            local_P = P // 2
            Lambda = Lambda[:local_P]
            B = B[:local_P]
        else:
            local_P = P
        self.local_P = local_P

        # Learnable complex eigenvalues (state dynamics)
        self.Lambda_param = nn.Parameter(torch.from_numpy(np.asarray(Lambda)))
        # Input-to-state projection matrix: shape (local_P, H_in)
        # We expand B (vector) to a matrix by repeating across the input dimension.
        self.B = nn.Parameter(torch.from_numpy(np.asarray(B)).unsqueeze(1).repeat(1, H_in))

        # Initialize state-to-output projection (complex) C_tilde.
        if C_init == "trunc_standard_normal":
            C_real = 0.1 * torch.randn(H_out, local_P)
            C_imag = 0.1 * torch.randn(H_out, local_P)
        elif C_init == "lecun_normal":
            fan_in = local_P
            std = math.sqrt(1.0 / fan_in)
            C_real = std * torch.randn(H_out, local_P)
            C_imag = std * torch.randn(H_out, local_P)
        else:
            raise NotImplementedError(f"C_init method {C_init} not implemented")
        C_complex = C_real.to(torch.complex64) + 1j * C_imag.to(torch.complex64)
        self.C_tilde = nn.Parameter(C_complex)  # shape: (H_out, local_P)

        # Feedthrough parameter D.
        # D is now always a matrix (H_out, H_in)
        self.D = nn.Parameter(0.1 * torch.randn(H_out, H_in))


        # Learnable log-step sizes (one per state dimension).
        log_dt_min = math.log(dt_min)
        log_dt_max = math.log(dt_max)
        log_steps = torch.rand(local_P) * (log_dt_max - log_dt_min) + log_dt_min
        self.log_step = nn.Parameter(log_steps)

        # Choose the discretization method.
        if discretization == "zoh":
            self.discretize_fn = discretize_zoh
        elif discretization == "dirac":
            self.discretize_fn = discretize_dirac
        elif discretization == "async":
            self.discretize_fn = discretize_async
        else:
            raise NotImplementedError(f"Discretization method {discretization} not implemented")
    def get_lambda_bar(self):
        step = self.step_rescale * torch.exp(self.log_step)
        # Compute discretized dynamics.
        Lambda_bar, gamma_bar = self.discretize_fn(self.Lambda_param, step, 0.05)
        self.lambda_bar = Lambda_bar
        return self.lambda_bar

    def step(self, u, dt, state=None, lesion_idx=None):
        """
        Perform a single time-step update.
        Args:
            u: (B, H_in) real tensor input.
            dt: scalar float integration timestep.
            state: (B, local_P) complex tensor of previous state; if None, initializes to zeros.
        Returns:
            output: (B, H_out) real tensor output.
            new_state: (B, local_P) complex tensor updated state.
        """
        B_size = u.shape[0]
        if state is None:
            state = torch.zeros(B_size, self.local_P, dtype=torch.complex64, device=u.device)
        # Compute learned step sizes.
        step = self.step_rescale * torch.exp(self.log_step)
        # Compute discretized dynamics.
        Lambda_bar, gamma_bar = self.discretize_fn(self.Lambda_param, step, dt)
        self.lambda_bar = Lambda_bar
        # Convert input to complex and project to state space.
        u_complex = u.to(torch.complex64)
        if lesion_idx:
            gamma_bar[lesion_idx] = 0
        Bu = gamma_bar.unsqueeze(0) * torch.matmul(self.B, u_complex.unsqueeze(-1)).squeeze(-1)
        # Update state.
        new_state = Lambda_bar.unsqueeze(0) * state + Bu  # shape: (B, local_P)
        # Compute output via state-to-output mapping.
        output_complex = torch.matmul(new_state, self.C_tilde.t())
        if self.conj_sym:
            output = 2 * output_complex.real
        else:
            output = output_complex.real
        # Add feedthrough term.
        # if self.H_in == self.H_out:
        #     output = output + u * self.D
        # else:
        output = output + torch.matmul(u, self.D.t())
        return output, new_state

class AC_SSM_stack(nn.Module):
    """
    Actor–Critic network that uses stacked SSM cells (S5SSMCell) as its recurrent core.

    Args:
        input_dimensions (int): Dimension of sensory input.
        action_dimensions (int): Number of possible actions.
        batch_size (int): Batch size.
        hidden_dim (int): Dimension of SSM cell output (for each cell).
        ssm_params (dict): Parameters for S5SSMCell. Expected keys include:
            'P', 'C_init', 'discretization', 'dt_min', 'dt_max', 'conj_sym', 'step_rescale'
        p_dropout (float): Dropout probability.
    """
    def __init__(self, input_dimensions, action_dimensions, hidden_dim, ssm_params, batch_size=1, p_dropout=0):
        super().__init__()
        self.input_d = input_dimensions
        self.action_d = action_dimensions
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim # Output dimension of each SSM cell

        # Instantiate the first SSM cell
        self.ssm_cell1 = S5SSMCell(
            H_in=input_dimensions,
            H_out=hidden_dim,
            P=ssm_params.get("P", 16),
            C_init=ssm_params.get("C_init", "trunc_standard_normal"),
            discretization=ssm_params.get("discretization", "zoh"),
            dt_min=ssm_params.get("dt_min", 0.001),
            dt_max=ssm_params.get("dt_max", 0.1),
            conj_sym=ssm_params.get("conj_sym", True),
            step_rescale=ssm_params.get("step_rescale", 1.0)
        )
        self.layer2 = ssm_params["layer2"]
        # Instantiate the second SSM cell.
        # The input dimension of the second cell is the output dimension of the first cell.
        if self.layer2:
            self.ssm_cell2 = S5SSMCell(
                H_in=hidden_dim, # Input is the output of the first SSM cell
                H_out=hidden_dim,
                P=ssm_params.get("P", 16), # Can use the same P or a different one
                C_init=ssm_params.get("C_init", "trunc_standard_normal"),
                discretization=ssm_params.get("discretization", "zoh"),
                dt_min=ssm_params.get("dt_min", 0.001),
                dt_max=ssm_params.get("dt_max", 0.1),
                conj_sym=ssm_params.get("conj_sym", True),
                step_rescale=ssm_params.get("step_rescale", 1.0)
            )

        # Actor and critic output layers.
        # They receive the output of the second SSM cell.
        self.actor = nn.Linear(hidden_dim, action_dimensions)
        self.critic = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(p=p_dropout)

        # Recurrent states are maintained here for each cell.
        self.hidden_state1 = None
        self.alpha = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.spiking = ssm_params.get("spike", False)
        if self.spiking:
            print("spiking SSM initialized.")
            self.ste = SpikeSTE.apply
        self.hidden_state2 = None


        # Add the missing attributes for training.
        self.saved_actions = []  # To store log probability and value pairs.
        self.rewards = []        # To store scalar rewards over time.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def forward(self, x, dt=0.05, lesion_idx=None):
        """
        Forward pass for one time step with stacked SSM cells.
        Args:
            x: (B, input_d) real tensor input.
            dt: scalar float integration timestep.
        Returns:
            policy: (B, action_d) softmax probabilities.
            value: (B, 1) state value.
            lin_act: (B, hidden_dim) raw SSM cell output from the *last* cell (before dropout), for logging.
        """
        # Pass input through the first SSM cell
        ssm_out1, self.hidden_state1 = self.ssm_cell1.step(x, dt, self.hidden_state1, lesion_idx)
        if self.spiking:
            ssm_out1 = self.ste(ssm_out1, self.alpha)
        # Pass the output of the first cell as input to the second SSM cell
        if self.layer2:
            ssm_out2, self.hidden_state2 = self.ssm_cell2.step(ssm_out1, dt, self.hidden_state2, lesion_idx)
            if self.spiking:
                ssm_out2 = self.ste(ssm_out2, self.alpha)
        else:
            ssm_out2 = ssm_out1
        # ssm_out2, self.hidden_state2 = ssm_out1, self.hidden_state1
        # The final linear activation comes from the output of the second cell
        lin_act = ssm_out2
        ssm_out2 = self.dropout(ssm_out2)

        policy = F.softmax(self.actor(ssm_out2), dim=1)
        value = self.critic(ssm_out2)
        if self. spiking:
            if self.layer2:
                return policy, value, lin_act, ssm_out1, lin_act
            return policy, value, lin_act, ssm_out1
        return policy, value, lin_act

    def reinit_hid(self):
        """
        Reinitialize the recurrent hidden states for both SSM cells.
        """
        self.hidden_state1 = None
        self.hidden_state2 = None


##########################################
# (Optional) Testing Code
##########################################
if __name__ == "__main__":
    # Quick test of AC_SSM with stacked cells on dummy data.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ssm_params = {
        "P": 32,
        "C_init": "trunc_standard_normal",
        "discretization": "zoh",
        "dt_min": 0.001,
        "dt_max": 0.1,
        "conj_sym": True,
        "step_rescale": 1.0
    }
    # Create an instance of the AC_SSM network with stacked cells.
    # Note: hidden_dim is the output dimension of each SSM cell.
    net = AC_SSM_stack(input_dimensions=2, action_dimensions=2, batch_size=1, hidden_dim=20, ssm_params=ssm_params, p_dropout=0.1).to(device)
    net.reinit_hid() # Corrected method name
    x = torch.randn(1, 2).to(device)
    policy, value, lin_act = net(x, dt=0.05)
    print("Policy:", policy)
    print("Value:", value)
    print("Output of last SSM cell (before dropout):", lin_act)

    # Test reinitialization
    net.reinit_hid()
    policy_reinit, value_reinit, lin_act_reinit = net(x, dt=0.05)
    print("\nAfter reinitialization:")
    print("Policy:", policy_reinit)
    print("Value:", value_reinit)
    print("Output of last SSM cell (before dropout):", lin_act_reinit)

    # Test the discount_rwds function
    rewards = np.array([1.0, 2.0, 3.0])
    discounted = discount_rwds(rewards, 0.9)
    print("\nDiscounted rewards:", discounted)

    # Test the finish_trial function (requires dummy data and optimizer)
    print("\nTesting finish_trial (dummy run):")
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    # Simulate a short trial
    net.rewards = [1.0, -0.5]
    # Need to populate saved_actions. Let's run a few steps.
    net.reinit_hid()
    x1 = torch.randn(1, 2).to(device)
    p1, v1, _ = net(x1, dt=0.05)
    action1 = select_action(net, p1, v1)[0] # just to add to saved_actions

    x2 = torch.randn(1, 2).to(device)
    p2, v2, _ = net(x2, dt=0.05)
    action2 = select_action(net, p2, v2)[0] # just to add to saved_actions

    try:
        p_loss, v_loss = finish_trial(net, 0.99, optimizer)
        print(f"Finish trial complete. Policy loss: {p_loss.item():.4f}, Value loss: {v_loss.item():.4f}")
        print("Rewards and saved_actions cleared:", len(net.rewards) == 0 and len(net.saved_actions) == 0)
    except Exception as e:
        print(f"Error during finish_trial test: {e}")
