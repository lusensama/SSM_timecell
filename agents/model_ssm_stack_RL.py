import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from collections import namedtuple
from agents import ssm_init
SavedAction = namedtuple('SavedAction', ['log_prob', 'value', 'policy'])

def discount_rwds(r, gamma):
    disc_rwds = np.zeros_like(r, dtype=float)
    running_add = 0.0
    for t in reversed(range(len(r))):
        running_add = running_add * gamma + r[t]
        disc_rwds[t] = running_add
    return disc_rwds

def select_action(model, policy_, value_):
    a = Categorical(policy_)
    action = a.sample()
    model.saved_actions.append(SavedAction(a.log_prob(action), value_, policy_))
    return action.item(), policy_.data[0], value_.squeeze().item()

def finish_trial(model, discount_factor, optimizer, scheduler=None,entropy_weight=0.5, **kwargs):
    returns_ = discount_rwds(np.asarray(model.rewards), gamma=discount_factor)
    saved_actions = model.saved_actions

    policy_losses = []
    value_losses = []
    entropy_terms = []

    returns_ = torch.Tensor(returns_).to(model.device)

    for (log_prob, value, policy_), r in zip(saved_actions, returns_):
        rpe = r - value.item()
        policy_losses.append(-log_prob * rpe)
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([[r]], device=model.device))
                             .unsqueeze(-1))
        entropy = -(policy_ * torch.log(policy_ + 1e-8)).sum()
        entropy_terms.append(entropy)
    optimizer.zero_grad()
    p_loss = torch.cat(policy_losses).sum()
    v_loss = torch.cat(value_losses).sum()
    if entropy_terms:
        entropy_bonus = -entropy_weight * torch.stack(entropy_terms).mean()
    else:
        entropy_bonus = 0
    total_loss = p_loss + v_loss + entropy_bonus

    total_loss.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    del model.rewards[:]
    del model.saved_actions[:]

    return p_loss, v_loss, total_loss

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

class S5SSMCell(nn.Module):
    def __init__(self, H_in, H_out, P,
                 C_init="trunc_standard_normal",
                 discretization="zoh",
                 dt_min=0.001, dt_max=0.1,
                 conj_sym=True, step_rescale=1.0,
                 init_mode="hippo", init_perturb_eps=0.1):
        super().__init__()
        self.H_in = H_in
        self.H_out = H_out
        self.P = P
        self.conj_sym = conj_sym
        self.step_rescale = step_rescale
        self.lambda_bar = None
        self.init_mode = init_mode

        Lambda, _, B, _, _ = ssm_init.make_DPLR_HiPPO(P)
        if self.conj_sym:
            local_P = P // 2
            Lambda = Lambda[:local_P]
            B = B[:local_P]
        else:
            local_P = P
        self.local_P = local_P

        lam_hippo = torch.from_numpy(np.asarray(Lambda)).to(torch.complex64)
        B_hippo = torch.from_numpy(np.asarray(B)).to(torch.complex64)
        lam, Bvec = self._init_lambda_B(lam_hippo, B_hippo, local_P,
                                        init_mode, init_perturb_eps, full_N=P)

        self.Lambda_param = nn.Parameter(lam)
        self.B = nn.Parameter(Bvec.unsqueeze(1).repeat(1, H_in))

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
    @staticmethod
    def _init_lambda_B(lam_hippo, B_hippo, local_P, init_mode, eps, full_N=None):
        if full_N is None:
            full_N = 2 * local_P
        def _cplx(re, im):
            return re.to(torch.complex64) + 1j * im.to(torch.complex64)

        s_im = lam_hippo.imag.std().clamp_min(1e-6)
        r0 = lam_hippo.real.mean().abs().clamp_min(1e-6)
        sBr = B_hippo.real.std().clamp_min(1e-6)
        sBi = B_hippo.imag.std().clamp_min(1e-6)

        if init_mode == "hippo":
            return lam_hippo.clone(), B_hippo.clone()

        if init_mode == "rand_complex":
            def _unit_disk(n):
                mag = torch.sqrt(torch.rand(n))
                ph = torch.rand(n) * (2.0 * math.pi)
                return mag * torch.cos(ph), mag * torch.sin(ph)
            lr, li = _unit_disk(local_P)
            br, bi = _unit_disk(local_P)
            return _cplx(lr, li), _cplx(br, bi)

        if init_mode == "spectrum_matched":
            perm = torch.randperm(local_P)
            im = lam_hippo.imag[perm]
            return _cplx(lam_hippo.real, im), B_hippo.clone()

        if init_mode == "freq_matched":
            re = -(r0 * (0.5 + torch.rand(local_P)))
            return _cplx(re, lam_hippo.imag), B_hippo.clone()

        if init_mode == "perturbed_hippo":
            scale = lam_hippo.abs().std().clamp_min(1e-6)
            noise = _cplx(torch.randn(local_P), torch.randn(local_P)) * (eps * scale)
            lam = lam_hippo + noise
            lam = _cplx(lam.real.clamp_max(-1e-4), lam.imag)
            return lam, B_hippo.clone()

        if init_mode == "alt_basis":
            fmax = float(lam_hippo.imag.abs().max())
            re = lam_hippo.real.clone()
            im = torch.linspace(0.0, fmax, local_P)
            return _cplx(re, im), B_hippo.clone()

        if init_mode == "s4d_lin":
            re = -0.5 * torch.ones(local_P)
            im = math.pi * torch.arange(local_P, dtype=torch.float32)
            return _cplx(re, im), B_hippo.clone()

        if init_mode == "s4d_inv":
            N = full_N
            n = torch.arange(local_P, dtype=torch.float32)
            re = -0.5 * torch.ones(local_P)
            im = (N / math.pi) * (N / (2.0 * n + 1.0) - 1.0)
            return _cplx(re, im), B_hippo.clone()

        if init_mode == "real_diagonal":
            re = -torch.logspace(-2, 0, local_P)
            im = torch.zeros(local_P)
            return _cplx(re, im), B_hippo.clone()

        raise NotImplementedError(f"init_mode '{init_mode}' not implemented")

    def get_lambda_bar(self):
        step = self.step_rescale * torch.exp(self.log_step)
        Lambda_bar, gamma_bar = self.discretize_fn(self.Lambda_param, step, 0.05)
        self.lambda_bar = Lambda_bar
        return self.lambda_bar

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
    def __init__(self, input_dimensions, action_dimensions, hidden_dim, ssm_params, batch_size=1, p_dropout=0):
        super().__init__()
        self.input_d = input_dimensions
        self.action_d = action_dimensions
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim

        self.ssm_cell1 = S5SSMCell(
            H_in=input_dimensions,
            H_out=hidden_dim,
            P=ssm_params.get("P", 16),
            C_init=ssm_params.get("C_init", "trunc_standard_normal"),
            discretization=ssm_params.get("discretization", "zoh"),
            dt_min=ssm_params.get("dt_min", 0.001),
            dt_max=ssm_params.get("dt_max", 0.1),
            conj_sym=ssm_params.get("conj_sym", True),
            step_rescale=ssm_params.get("step_rescale", 1.0),
            init_mode=ssm_params.get("init_mode", "hippo"),
            init_perturb_eps=ssm_params.get("init_perturb_eps", 0.1)
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
                step_rescale=ssm_params.get("step_rescale", 1.0),
                init_mode=ssm_params.get("init_mode", "hippo"),
                init_perturb_eps=ssm_params.get("init_perturb_eps", 0.1)
            )

        self.actor = nn.Linear(hidden_dim, action_dimensions)
        self.critic = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(p=p_dropout)

        self.hidden_state1 = None
        self.alpha = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.spiking = ssm_params.get("spike", False)
        if self.spiking:
            print("spiking SSM initialized.")
            self.ste = SpikeSTE.apply
        self.hidden_state2 = None

        self.saved_actions = []
        self.rewards = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x, dt=0.05, lesion_idx=None):
        ssm_out1, self.hidden_state1 = self.ssm_cell1.step(x, dt, self.hidden_state1, lesion_idx)
        if self.spiking:
            ssm_out1 = self.ste(ssm_out1, self.alpha)
        if self.layer2:
            ssm_out2, self.hidden_state2 = self.ssm_cell2.step(ssm_out1, dt, self.hidden_state2, lesion_idx)
            if self.spiking:
                ssm_out2 = self.ste(ssm_out2, self.alpha)
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

if __name__ == "__main__":
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
    net = AC_SSM_stack(input_dimensions=2, action_dimensions=2, batch_size=1, hidden_dim=20, ssm_params=ssm_params, p_dropout=0.1).to(device)
    net.reinit_hid()
    x = torch.randn(1, 2).to(device)
    policy, value, lin_act = net(x, dt=0.05)
    print("Policy:", policy)
    print("Value:", value)
    print("Output of last SSM cell (before dropout):", lin_act)

    net.reinit_hid()
    policy_reinit, value_reinit, lin_act_reinit = net(x, dt=0.05)
    print("\nAfter reinitialization:")
    print("Policy:", policy_reinit)
    print("Value:", value_reinit)
    print("Output of last SSM cell (before dropout):", lin_act_reinit)

    rewards = np.array([1.0, 2.0, 3.0])
    discounted = discount_rwds(rewards, 0.9)
    print("\nDiscounted rewards:", discounted)

    print("\nTesting finish_trial (dummy run):")
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    net.rewards = [1.0, -0.5]
    net.reinit_hid()
    x1 = torch.randn(1, 2).to(device)
    p1, v1, _ = net(x1, dt=0.05)
    action1 = select_action(net, p1, v1)[0]

    x2 = torch.randn(1, 2).to(device)
    p2, v2, _ = net(x2, dt=0.05)
    action2 = select_action(net, p2, v2)[0]

    try:
        p_loss, v_loss = finish_trial(net, 0.99, optimizer)
        print(f"Finish trial complete. Policy loss: {p_loss.item():.4f}, Value loss: {v_loss.item():.4f}")
        print("Rewards and saved_actions cleared:", len(net.rewards) == 0 and len(net.saved_actions) == 0)
    except Exception as e:
        print(f"Error during finish_trial test: {e}")

