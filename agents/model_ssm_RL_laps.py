"""
model_s5ssm.py

This module implements an SSM-based actor–critic network with HIPPO-based initialization,
extended with utilities tailored for lap-counting experiments.
It contains:
 - Discretization and HIPPO initialization utilities.
 - S5SSMCell: a single-step state-space model cell.
 - AC_SSM: an actor–critic network that uses S5SSMCell as its recurrent core,
           modified to stack two SSM cells.

Differences vs `agents/model_ssm_stack_RL.py` (generic variant):
- JAX interop: fetches HIPPO params via `agents.ssm_init` and explicitly uses `jax.device_get`.
- Spiking mode: includes an LIF surrogate gradient (LIFSpike) with membrane state `mem` and `threshold`.
- Training helper: provides `finish_run` with normalized returns and entropy weighting.
- Probe utilities: includes `add_probe`, `forward_linear_probe`, and `finish_readout` for linear probes.
- Device specifics: some tensors are created on `'cuda:0'` (may require adjustment on CPU-only setups).
- Forward API: returns `(policy, value, lin_act)`; under spiking/layer2 may return extra intermediates.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from collections import namedtuple
from agents import ssm_init
import jax
##########################################
# HIPPO Initialization & Discretization Utilities
##########################################
# Named tuple to store log probability and state value for each action.
SavedAction = namedtuple('SavedAction', ['log_prob', 'value', 'policy'])


def discount_rwds(r, gamma):
    """
    Compute discounted rewards-to-go for each timestep.

    Args:
        r (np.array): An array of rewards for each timestep of an episode.
        gamma (float): The discount factor (e.g., 0.99).

    Returns:
        np.array: The array of discounted rewards (returns).
    """
    disc_rwds = np.zeros_like(r, dtype=float)
    running_add = 0.0
    # Iterate backwards through the rewards to calculate the discounted return at each step
    for t in reversed(range(len(r))):
        running_add = running_add * gamma + r[t]
        disc_rwds[t] = running_add
    return disc_rwds


def select_action(model, policy_, value_):
    """
    Samples an action from the policy distribution, stores action info, and returns the action.

    Args:
        model: The actor–critic model. Must have a `saved_actions` list attribute.
        policy_ (torch.Tensor): A tensor from the model's actor head, representing action probabilities.
        value_ (torch.Tensor): A tensor from the model's critic head, representing the state value estimate.

    Returns:
        action (int): The sampled action.
        policy_info (torch.Tensor): The raw policy tensor for logging.
        value_info (float): The state value as a float for logging.
    """
    # Create a categorical distribution over the list of probabilities of actions
    m = Categorical(policy_)
    # Sample an action from the distribution
    action = m.sample()

    # Save the log probability of the sampled action, the state value, and the policy
    model.saved_actions.append(SavedAction(m.log_prob(action), value_, policy_))

    return action.item(), policy_.data, value_.squeeze().item()


def finish_run(model, discount_factor, optimizer, scheduler=None, value_loss_weight=0.5, entropy_weight=0.01):
    """
    Completes a training trial (episode) by computing losses and updating model weights.
    This function implements a full Advantage Actor-Critic (A2C) update step.

    Returns:
        Tuple of (policy_loss, value_loss, total_loss) as floats for logging.
    """
    saved_actions = model.saved_actions

    # 1. Calculate the discounted rewards (returns) for the entire episode.
    returns = discount_rwds(np.asarray(model.rewards), gamma=discount_factor)
    returns = torch.tensor(returns, dtype=torch.float32).to(model.device)

    # Normalize returns for better stability (optional but recommended)
    returns = (returns - returns.mean()) / (returns.std() + 1e-6)

    policy_losses = []
    value_losses = []
    entropy_bonuses = []

    # 2. Iterate through saved actions and calculated returns to compute losses
    for (log_prob, value, policy), R in zip(saved_actions, returns):
        # 3. Calculate the Advantage: A(s,a) = R - V(s)
        # R is the discounted return (our target for the critic)
        # value is the critic's estimate V(s)
        advantage = R - value

        # 4. Calculate Actor Loss (Policy Gradient)
        # We use .detach() on the advantage so that we don't backpropagate gradients
        # from the policy loss into the critic. The critic is trained separately.
        policy_losses.append(-log_prob * advantage.detach())

        # 5. Calculate Critic Loss
        # The critic is updated to minimize the error between its estimate V(s) and the actual return R.
        # We use Mean Squared Error, a common choice for value function approximation.
        value_losses.append(F.mse_loss(value.squeeze(), R))

        # 6. Calculate Entropy Bonus
        # This encourages exploration by penalizing overly confident (low-entropy) policies.
        entropy = -(policy * torch.log(policy + 1e-8)).sum()
        entropy_bonuses.append(entropy)

    # 7. Sum up the losses and perform the backward pass
    optimizer.zero_grad()

    # Combine losses:
    # - Sum of policy losses
    # - Sum of value losses (weighted)
    # - Negative of entropy bonus (we want to maximize entropy, so we subtract the bonus)
    policy_loss = torch.stack(policy_losses).sum()
    value_loss = torch.stack(value_losses).sum()
    entropy_bonus = -entropy_weight * torch.stack(entropy_bonuses).mean()

    total_loss = policy_loss + value_loss_weight * value_loss + entropy_bonus

    # Backpropagate and update the weights
    total_loss.backward(retain_graph=True)

    # Clip gradients to prevent them from exploding, a common practice in RL
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    # Clear the episode's rewards and saved actions
    del model.rewards[:]
    del model.saved_actions[:]

    return policy_loss.item(), value_loss.item(), total_loss.item()

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


class LIFSpike(torch.autograd.Function):
    """
    A single function that implements the Leaky Integrate-and-Fire (LIF) neuron
    dynamics and the surrogate gradient for backpropagation.
    The forward pass computes the neuron's state update (integrate, fire, reset).
    The backward pass computes the surrogate gradient.
    This version is designed to prevent in-place modification errors in recurrent settings.
    """

    @staticmethod
    def forward(ctx, input_current, mem_in, threshold, alpha):
        # 1. Integrate input current to get post-integration potential
        mem_after_integration = mem_in + input_current

        # 2. Save the post-integration potential for the backward pass.
        # This is the value that the spiking is based on.
        ctx.save_for_backward(mem_after_integration - threshold, alpha)

        # 3. Spike generation based on post-integration potential
        spikes = (mem_after_integration > threshold).float()

        # 4. Reset membrane potential for neurons that spiked
        mem_out = mem_after_integration * (1 - spikes)

        return spikes, mem_out

    @staticmethod
    def backward(ctx, grad_spikes, grad_mem):
        # Retrieve saved tensors: u = (mem_after_integration - threshold)
        u, alpha = ctx.saved_tensors

        # Calculate surrogate gradient
        sigma = torch.sigmoid(alpha * u)
        surrogate_grad = alpha * sigma * (1 - sigma)

        # The gradient w.r.t. the spikes is scaled by the surrogate gradient
        grad_from_spikes = grad_spikes * surrogate_grad

        # Recompute spikes to mask the gradient from the membrane potential reset
        # For neurons that spiked, the gradient from the next membrane potential should be zero.
        spikes = (u > 0).float()
        grad_from_mem = grad_mem * (1 - spikes)

        # The total gradient w.r.t. the post-integration potential is the sum of the two paths
        total_grad_u = grad_from_spikes + grad_from_mem

        # The gradient flows back equally to the input current and the previous membrane state
        grad_input_current = total_grad_u
        grad_prev_mem = total_grad_u

        # Gradient for the learnable parameter alpha
        grad_alpha = (grad_spikes * u * sigma * (1 - sigma)).sum()

        # Return gradients for each input of the forward function
        return grad_input_current, grad_prev_mem, None, grad_alpha
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
                 conj_sym=True, step_rescale=1.0,
                 tau_m=20.0, v_reset=0.0, v_thresh=1.0, input_scale=1.0):
        super().__init__()
        self.H_in = H_in
        self.H_out = H_out
        self.P = P
        self.conj_sym = conj_sym
        self.step_rescale = step_rescale
        self.lambda_bar = None

        # Initialize HIPPO components for a single block of size P.
        # Lambda, _, B, _, _ = make_DPLR_HiPPO(P)
        # Fetch HIPPO parameters (originally JAX arrays).
        Lambda, _, B, _, _ = ssm_init.make_DPLR_HiPPO(P)
        # Move them to host (NumPy) before any np.asarray or torch conversion:
        Lambda = np.array(
jax.device_get(Lambda), copy=True)
        B      = np.array(jax.device_get(B), copy=True)

        if self.conj_sym:
            local_P = P // 2
            Lambda = Lambda[:local_P]
            B      = B[:local_P]
        else:
            local_P = P
        self.local_P = local_P

        # Learnable real eigenvalues (state dynamics); Lambda is now a host NumPy array
        self.Lambda_param = nn.Parameter(torch.from_numpy(Lambda))

        # Input-to-state projection matrix: shape (local_P, H_in)
        # We expand B (vector) to a matrix by repeating across the input dimension.
        self.B = nn.Parameter(torch.from_numpy(B).unsqueeze(1).repeat(1, H_in))

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
    def __init__(self, input_dimensions, action_dimensions, hidden_dim, ssm_params, batch_size=1, p_dropout=0, threshold=0.5):
        super().__init__()
        self.input_d = input_dimensions
        self.action_d = action_dimensions
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim # Output dimension of each SSM cell
        self.mem = torch.zeros(hidden_dim, device='cuda:0')
        self.threshold = torch.tensor([threshold], device='cuda:0')
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
        self.linear_probe = nn.Sequential(
            nn.Linear(hidden_dim, 100),   # first layer
            nn.GELU(),        # non‑linearity (swap for GELU, SiLU, etc. if you prefer)
            nn.Linear(100, 4)
            # nn.Linear(100, ssm_params.get("lap_count", 4))  # second layer → logits over laps
        )
        # self.linear_probe = nn.Linear(hidden_dim,ssm_params.get("lap_count", 4))

        # Recurrent states are maintained here for each cell.
        self.hidden_state1 = None
        self.alpha = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.spiking = ssm_params.get("spike", False)
        if self.spiking:
            print("spiking SSM initialized.")
            # self.ste = SpikeSTE.apply
            self.ste = LIFSpike.apply
        self.hidden_state2 = None


        # Add the missing attributes for training.
        self.saved_actions = []  # To store log probability and value pairs.
        self.rewards = []        # To store scalar rewards over time.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def add_probe(self):
        self.linear_probe = nn.LSTMCell(self.hidden_dim, 4, device=self.device)
    def forward_linear_probe(self, x, dt=0.05, lesion_idx=None):
        """
        New forward pass for training a linear probe on top of the frozen SSM stack.
        It computes the SSM representation in no_grad mode and passes it to a new trainable linear layer.
        Args:
            x: (B, input_d) real tensor input.
            dt: scalar float integration timestep.
        Returns:
            probe_output: (B, 4) raw logits from the new linear probe layer.
        """
        # Step 1: Get the SSM output with frozen weights.
        # The torch.no_grad() context manager ensures that all operations
        # inside it are not tracked for gradient computation, effectively
        # freezing the ssm_cell layers for this pass.
        with torch.no_grad():
            ssm_out1, self.hidden_state1 = self.ssm_cell1.step(x, dt, self.hidden_state1, lesion_idx)
            if self.spiking:
                ssm_out1 = self.ste(ssm_out1, self.alpha)

            if self.layer2:
                ssm_out2, self.hidden_state2 = self.ssm_cell2.step(ssm_out1, dt, self.hidden_state2, lesion_idx)
                if self.spiking:
                    ssm_out2 = self.ste(ssm_out2, self.alpha)
            else:
                ssm_out2 = ssm_out1

            # The raw output of the final SSM cell is used as the representation to probe.
            lin_act = ssm_out2.detach()  # Use .detach() as an extra safeguard

        # Step 2: Pass the representation through the new linear layer.
        # This operation is *outside* the no_grad() context, so PyTorch will
        # compute gradients for self.linear_probe, allowing it to be trained.
        probe_output= self.linear_probe(lin_act)
        # probe_output = self.linear_probe(self.hidden_state1.real)

        return probe_output

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
            # ssm_out1 = self.ste(ssm_out1, self.alpha)
            ssm_out1, self.mem = self.ste(ssm_out1, self.mem.detach(), self.threshold, self.alpha)
        # Pass the output of the first cell as input to the second SSM cell
        if self.layer2:
            ssm_out2, self.hidden_state2 = self.ssm_cell2.step(ssm_out1, dt, self.hidden_state2, lesion_idx)
            if self.spiking:
                # ssm_out2 = self.ste(ssm_out2, self.alpha)
                ssm_out1, self.mem = self.ste(ssm_out1, self.mem.detach(), self.threshold, self.alpha)
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

def finish_readout(model, probe_output, label, optimizer, lap_ending):
    """
    Computes the loss, performs one step of backpropagation, and updates the weights
    for the linear probe layer.

    Args:
        model (torch.nn.Module): The model containing the linear_probe.
        probe_output (torch.Tensor): Raw logits from the forward_linear_probe function. Shape: (B, num_classes).
        label (torch.Tensor): Ground truth labels. Shape: (B,).
        optimizer (torch.optim.Optimizer): The optimizer configured to train *only* the
                                           linear_probe's parameters.
        loss_fn (torch.nn.Module): The loss function, e.g., nn.CrossEntropyLoss.

    Returns:
        pred (torch.Tensor): The predicted class indices for the batch.
        total_loss (torch.Tensor): The scalar loss value for this step.
    """
    # Step 1: Calculate the loss between the probe's output and the true labels.
    # For classification, CrossEntropyLoss is standard and expects raw logits.
    # loss_fn = torch.nn.MSELoss()
    label = torch.tensor([label], dtype=torch.float32, device='cuda')
    # probe_output = torch.nn.Softmax()(probe_output)
    weight = 100 if lap_ending else 0.001
    total_loss = F.mse_loss(probe_output, label, reduction='sum')*weight
    # total_loss = loss_fn(probe_output, label)

    # Step 2: Zero out gradients from the previous training step.
    # This is essential to prevent gradients from accumulating.
    optimizer.zero_grad()

    # Step 3: Perform backpropagation.
    # PyTorch automatically computes gradients for all parameters that have
    # `requires_grad=True` and were part of the computation graph.
    # Because of the torch.no_grad() in the forward pass and how the optimizer
    # was set up, this will only compute gradients for `model.linear_probe`.
    total_loss.backward()

    # Step 4 (Optional but Recommended): Clip gradients to prevent exploding gradients.
    # This is applied to the parameters of the layer being trained.
    torch.nn.utils.clip_grad_norm_(model.linear_probe.parameters(), max_norm=1.0)

    # Step 5: Update the weights of the linear probe.
    # The optimizer uses the computed gradients to adjust the parameters.
    optimizer.step()

    # Get the model's prediction by finding the class with the highest logit.
    with torch.no_grad():
        pred = torch.argmax(probe_output, dim=1)

    return pred, total_loss
##########################################
# (Optional) Testing Code
##########################################
# if __name__ == "__main__":
#     # Quick test of AC_SSM with stacked cells on dummy data.
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     ssm_params = {
#         "P": 32,
#         "C_init": "trunc_standard_normal",
#         "discretization": "zoh",
#         "dt_min": 0.001,
#         "dt_max": 0.1,
#         "conj_sym": True,
#         "step_rescale": 1.0
#     }
#     # Create an instance of the AC_SSM network with stacked cells.
#     # Note: hidden_dim is the output dimension of each SSM cell.
#     net = AC_SSM_