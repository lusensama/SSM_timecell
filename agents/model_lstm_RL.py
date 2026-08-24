import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.model_ssm_stack_RL import SpikeSTE

HIDDEN_TYPES = ("lstm",)

class AC_RNN(nn.Module):

    def __init__(self, input_dimensions, action_dimensions, hidden_dim,
                 hidden_type="lstm", batch_size=1, p_dropout=0.0,
                 spike=True, rnn_spike=False):
        super().__init__()

        if hidden_type not in HIDDEN_TYPES:
            raise ValueError(
                f"hidden_type must be one of {HIDDEN_TYPES}, got {hidden_type!r}"
            )

        self.input_d = input_dimensions
        self.action_d = action_dimensions
        self.hidden_dim = hidden_dim
        self.hidden_type = hidden_type
        self.batch_size = batch_size
        self.spiking = spike
        self.rnn_spike = rnn_spike

        self.cell1 = nn.LSTMCell(input_dimensions, hidden_dim)

        self.readout = nn.Linear(hidden_dim, hidden_dim)

        self.actor = nn.Linear(hidden_dim, action_dimensions)
        self.critic = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(p=p_dropout)

        self.alpha = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        if self.rnn_spike:
            print(f"spiking {hidden_type.upper()} initialized.")
            self.ste = SpikeSTE.apply

        self.hx = []
        self.cx = []
        self.state_source = "h"

        self.saved_actions = []
        self.rewards = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.reinit_hid()

    @property
    def n_cells(self):
        return 1

    @property
    def hidden_state1(self):
        return self._state_of(0)

    @property
    def hidden_state2(self):
        return self._state_of(1)

    def _state_of(self, i):
        if i >= self.n_cells:
            return None
        if self.state_source == "c":
            return self.cx[i]
        return self.hx[i]

    def reinit_hid(self):
        dev = next(self.parameters()).device
        self.hx = []
        self.cx = []
        for _ in range(self.n_cells):
            self.hx.append(torch.zeros(self.batch_size, self.hidden_dim, device=dev))
            self.cx.append(torch.zeros(self.batch_size, self.hidden_dim, device=dev))

    def _step_cell(self, cell, x, i, lesion_idx=None):
        hx = self.hx[i]
        cx = self.cx[i]
        if lesion_idx is not None:
            hx = hx.clone().detach()
            hx[:, lesion_idx] = 0
            if cx is not None:
                cx = cx.clone().detach()
                cx[:, lesion_idx] = 0

        h_new, c_new = cell(x, (hx, cx))
        self.hx[i] = h_new
        self.cx[i] = c_new
        return h_new

    def forward(self, x, dt=0.05, lesion_idx=None):
        out1 = self._step_cell(self.cell1, x, 0, lesion_idx)
        if self.rnn_spike:
            out1 = self.ste(out1, self.alpha)

        lin_act = F.relu(self.readout(out1))
        head_in = self.dropout(lin_act)

        policy = F.softmax(self.actor(head_in), dim=1)
        value = self.critic(head_in)

        if self.spiking:
            return policy, value, lin_act, out1
        return policy, value, lin_act

def freeze_rnn_params(net, freeze_lambda=False, freeze_B=False):
    cells = [("cell1", net.cell1)]

    for name, cell in cells:
        if freeze_lambda:
            cell.weight_hh.requires_grad_(False)
            if cell.bias_hh is not None:
                cell.bias_hh.requires_grad_(False)
            print(f"Froze recurrent weights (A analog) for {name}.")
        if freeze_B:
            cell.weight_ih.requires_grad_(False)
            if cell.bias_ih is not None:
                cell.bias_ih.requires_grad_(False)
            print(f"Froze input projection (B analog) for {name}.")

if __name__ == "__main__":
    for htype in HIDDEN_TYPES:
        for spike in [False, True]:
            net = AC_RNN(input_dimensions=3, action_dimensions=3, hidden_dim=8,
                         hidden_type=htype, spike=spike,
                         p_dropout=0.1)
            net.reinit_hid()
            out = net.forward(torch.zeros(1, 3))
            expected = 4 if spike else 3
            assert len(out) == expected, (htype, spike, len(out))
            assert out[0].shape == (1, 3) and out[1].shape == (1, 1)
        print(f"{htype:7s} arity OK")

    net = AC_RNN(3, 3, 8, hidden_type="lstm")
    net.reinit_hid()
    net.forward(torch.ones(1, 3))
    h1 = net.hx[0].clone()
    net.forward(torch.ones(1, 3))
    assert not torch.allclose(h1, net.hx[0]), "recurrent state is not evolving"
    net.reinit_hid()
    assert torch.count_nonzero(net.hx[0]) == 0

    freeze_rnn_params(net, freeze_lambda=True, freeze_B=True)
    trainable = {n for n, p in net.named_parameters() if p.requires_grad}
    assert not any(n.startswith("cell1.") for n in trainable), trainable
    assert {"readout.weight", "actor.weight", "critic.weight"} <= trainable
    print("state + freeze OK; trainable after freeze:", sorted(trainable))

