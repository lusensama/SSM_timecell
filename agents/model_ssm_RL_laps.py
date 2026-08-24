import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from collections import namedtuple
from agents import ssm_init
import jax
SavedAction = namedtuple('SavedAction', ['log_prob', 'value', 'policy'])

def discount_rwds(r, gamma):
    disc_rwds = np.zeros_like(r, dtype=float)
    running_add = 0.0
    for t in reversed(range(len(r))):
        running_add = running_add * gamma + r[t]
        disc_rwds[t] = running_add
    return disc_rwds

def select_action(model, policy_, value_):
    m = Categorical(policy_)
    action = m.sample()

    model.saved_actions.append(SavedAction(m.log_prob(action), value_, policy_))

    return action.item(), policy_.data, value_.squeeze().item()

def finish_run(model, discount_factor, optimizer, scheduler=None, value_loss_weight=0.5, entropy_weight=0.01):
    saved_actions = model.saved_actions

    returns = discount_rwds(np.asarray(model.rewards), gamma=discount_factor)
    returns = torch.tensor(returns, dtype=torch.float32).to(model.device)

    returns = (returns - returns.mean()) / (returns.std() + 1e-6)

    policy_losses = []
    value_losses = []
    entropy_bonuses = []

    for (log_prob, value, policy), R in zip(saved_actions, returns):
        advantage = R - value

        policy_losses.append(-log_prob * advantage.detach())

        value_losses.append(F.mse_loss(value.squeeze(), R))

        entropy = -(policy * torch.log(policy + 1e-8)).sum()
        entropy_bonuses.append(entropy)

    optimizer.zero_grad()

    policy_loss = torch.stack(policy_losses).sum()
    value_loss = torch.stack(value_losses).sum()
    entropy_bonus = -entropy_weight * torch.stack(entropy_bonuses).mean()

    total_loss = policy_loss + value_loss_weight * value_loss + entropy_bonus

    total_loss.backward(retain_graph=True)

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    del model.rewards[:]
    del model.saved_actions[:]

    return policy_loss.item(), value_loss.item(), total_loss.item()

def discretize_zoh(Lambda, step_delta, time_delta):
    Identity = torch.ones(Lambda.shape, dtype=Lambda.dtype, device=Lambda.device)
    Delta = step_delta * time_delta
    Lambda_bar = torch.exp(Lambda * Delta)
    gamma_bar = (1.0 / Lambda) * (Lambda_bar - Identity)
    return Lambda_bar, gamma_bar

def discretize_dirac(Lambda, step_delta, time_delta):
    Delta = step_delta * time_delta
    Lambda_bar = torch.exp(Lambda * Delta)
    gamma_bar = torch.tensor(1.0, dtype=Lambda.dtype, device=Lambda.device)
    return Lambda_bar, gamma_bar

def discretize_async(Lambda, step_delta, time_delta):
    Identity = torch.ones(Lambda.shape, dtype=Lambda.dtype, device=Lambda.device)
    Lambda_bar = torch.exp(Lambda * step_delta * time_delta)
    gamma_bar = (1.0 / Lambda) * (torch.exp(Lambda * step_delta) - Identity)
    return Lambda_bar, gamma_bar

def make_HiPPO(N):
    n = torch.arange(N, dtype=torch.float32)
    P_vec = torch.sqrt(1 + 2 * n)
    A = torch.tril(P_vec[:, None] * P_vec[None, :]) - torch.diag(n)
    return -A

def make_NPLR_HiPPO(N):
    n = torch.arange(N, dtype=torch.float32)
    A = make_HiPPO(N)
    P_vec = torch.sqrt(n + 0.5)
    B = torch.sqrt(2 * n + 1.0)
    return A, P_vec, B

def make_DPLR_HiPPO(N):
    A, P_vec, B = make_NPLR_HiPPO(N)
    S = A + torch.outer(P_vec, P_vec)
    S_diag = torch.diag(S)
    Lambda_real = S_diag.mean() * torch.ones_like(S_diag)

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
        sigma = torch.sigmoid(alpha * x)
        surrogate_grad = alpha * sigma * (1 - sigma)
        return grad_output * surrogate_grad, (grad_output * sigma *(1-sigma) ).sum()

class LIFSpike(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_current, mem_in, threshold, alpha):
        mem_after_integration = mem_in + input_current

        ctx.save_for_backward(mem_after_integration - threshold, alpha)

        spikes = (mem_after_integration > threshold).float()

        mem_out = mem_after_integration * (1 - spikes)

        return spikes, mem_out

    @staticmethod
    def backward(ctx, grad_spikes, grad_mem):
        u, alpha = ctx.saved_tensors

        sigma = torch.sigmoid(alpha * u)
        surrogate_grad = alpha * sigma * (1 - sigma)

        grad_from_spikes = grad_spikes * surrogate_grad

        spikes = (u > 0).float()
        grad_from_mem = grad_mem * (1 - spikes)

        total_grad_u = grad_from_spikes + grad_from_mem

        grad_input_current = total_grad_u
        grad_prev_mem = total_grad_u

        grad_alpha = (grad_spikes * u * sigma * (1 - sigma)).sum()

        return grad_input_current, grad_prev_mem, None, grad_alpha

class S5SSMCell(nn.Module):
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

        Lambda, _, B, _, _ = ssm_init.make_DPLR_HiPPO(P)
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

        self.Lambda_param = nn.Parameter(torch.from_numpy(Lambda))

        self.B = nn.Parameter(torch.from_numpy(B).unsqueeze(1).repeat(1, H_in))

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
        self.C_tilde = nn.Parameter(C_complex)

        self.D = nn.Parameter(0.1 * torch.randn(H_out, H_in))

        log_dt_min = math.log(dt_min)
        log_dt_max = math.log(dt_max)
        log_steps = torch.rand(local_P) * (log_dt_max - log_dt_min) + log_dt_min
        self.log_step = nn.Parameter(log_steps)

        if discretization == "zoh":
            self.discretize_fn = discretize_zoh
        elif discretization == "dirac":
            self.discretize_fn = discretize_dirac
        elif discretization == "async":
            self.discretize_fn = discretize_async
        else:
            raise NotImplementedError(f"Discretization method {discretization} not implemented")

    def step(self, u, dt, state=None, lesion_idx=None):
        B_size = u.shape[0]
        if state is None:
            state = torch.zeros(B_size, self.local_P, dtype=torch.complex64, device=u.device)
        step = self.step_rescale * torch.exp(self.log_step)
        Lambda_bar, gamma_bar = self.discretize_fn(self.Lambda_param, step, dt)
        self.lambda_bar = Lambda_bar
        u_complex = u.to(torch.complex64)
        if lesion_idx:
            gamma_bar[lesion_idx] = 0
        Bu = gamma_bar.unsqueeze(0) * torch.matmul(self.B, u_complex.unsqueeze(-1)).squeeze(-1)
        new_state = Lambda_bar.unsqueeze(0) * state + Bu
        output_complex = torch.matmul(new_state, self.C_tilde.t())
        if self.conj_sym:
            output = 2 * output_complex.real
        else:
            output = output_complex.real
        output = output + torch.matmul(u, self.D.t())
        return output, new_state

class AC_SSM_stack(nn.Module):
    def __init__(self, input_dimensions, action_dimensions, hidden_dim, ssm_params, batch_size=1, p_dropout=0, threshold=0.5):
        super().__init__()
        self.input_d = input_dimensions
        self.action_d = action_dimensions
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mem = torch.zeros(hidden_dim, device=self.device)
        self.threshold = torch.tensor([threshold], device=self.device)
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
        if self.layer2:
            self.ssm_cell2 = S5SSMCell(
                H_in=hidden_dim,
                H_out=hidden_dim,
                P=ssm_params.get("P", 16),
                C_init=ssm_params.get("C_init", "trunc_standard_normal"),
                discretization=ssm_params.get("discretization", "zoh"),
                dt_min=ssm_params.get("dt_min", 0.001),
                dt_max=ssm_params.get("dt_max", 0.1),
                conj_sym=ssm_params.get("conj_sym", True),
                step_rescale=ssm_params.get("step_rescale", 1.0)
            )

        self.actor = nn.Linear(hidden_dim, action_dimensions)
        self.critic = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(p=p_dropout)
        self.linear_probe = nn.Sequential(
            nn.Linear(hidden_dim, 100),
            nn.GELU(),
            nn.Linear(100, 4)
        )

        self.hidden_state1 = None
        self.alpha = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.spiking = ssm_params.get("spike", False)
        if self.spiking:
            print("spiking SSM initialized.")
            self.ste = LIFSpike.apply
        self.hidden_state2 = None

        self.saved_actions = []
        self.rewards = []
    def add_probe(self):
        self.linear_probe = nn.LSTMCell(self.hidden_dim, 4, device=self.device)
    def forward_linear_probe(self, x, dt=0.05, lesion_idx=None):
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

            lin_act = ssm_out2.detach()

        probe_output= self.linear_probe(lin_act)

        return probe_output

    def forward(self, x, dt=0.05, lesion_idx=None):
        ssm_out1, self.hidden_state1 = self.ssm_cell1.step(x, dt, self.hidden_state1, lesion_idx)
        if self.spiking:
            ssm_out1, self.mem = self.ste(ssm_out1, self.mem.detach(), self.threshold, self.alpha)
        if self.layer2:
            ssm_out2, self.hidden_state2 = self.ssm_cell2.step(ssm_out1, dt, self.hidden_state2, lesion_idx)
            if self.spiking:
                ssm_out1, self.mem = self.ste(ssm_out1, self.mem.detach(), self.threshold, self.alpha)
        else:
            ssm_out2 = ssm_out1
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
        self.hidden_state1 = None
        self.hidden_state2 = None

def finish_readout(model, probe_output, label, optimizer, lap_ending):
    label = torch.tensor([label], dtype=torch.float32, device='cuda')
    weight = 100 if lap_ending else 0.001
    total_loss = F.mse_loss(probe_output, label, reduction='sum')*weight

    optimizer.zero_grad()

    total_loss.backward()

    torch.nn.utils.clip_grad_norm_(model.linear_probe.parameters(), max_norm=1.0)

    optimizer.step()

    with torch.no_grad():
        pred = torch.argmax(probe_output, dim=1)

    return pred, total_loss

