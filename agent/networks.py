"""Actor and critic networks."""
import torch
import torch.nn as nn
from torch.distributions import Normal

from .config import state_dim, action_dim, DEFAULT_POSE, JOINT_MIN, JOINT_MAX


def _obs_normalizer():
    """
    Fixed (not learned) shift/scale per state index so every input is roughly in [-3, 3].
    Raw positions/velocities are in hundreds of pixels while sin/cos are O(1); without this the
    network is dominated by the large inputs and learns very slowly.
    """
    offset = torch.zeros(state_dim)
    scale = torch.ones(state_dim)
    for i in (33, 37, 41, 45):            # x positions relative to spawn
        scale[i] = 100.0
    for i in (35, 36, 39, 40, 43, 44, 47):  # linear velocities
        scale[i] = 400.0
    offset[34], scale[34] = 160.0, 60.0   # hip y
    offset[38], scale[38] = 20.0, 40.0    # left foot y
    offset[42], scale[42] = 20.0, 40.0    # right foot y
    offset[46], scale[46] = 216.0, 60.0   # chest y
    return offset, scale


class ObsNorm(nn.Module):
    def __init__(self):
        super().__init__()
        offset, scale = _obs_normalizer()
        self.register_buffer("offset", offset)
        self.register_buffer("scale", scale)

    def forward(self, state):
        return ((state - self.offset) / self.scale).clamp(-10.0, 10.0)


def _mlp(in_dim, out_dim, out_gain):
    net = nn.Sequential(
        nn.Linear(in_dim, 256), nn.Tanh(),
        nn.Linear(256, 256), nn.Tanh(),
        nn.Linear(256, out_dim),
    )
    for layer in (net[0], net[2]):
        nn.init.orthogonal_(layer.weight, gain=2 ** 0.5)
        nn.init.constant_(layer.bias, 0.0)
    nn.init.orthogonal_(net[4].weight, gain=out_gain)
    nn.init.constant_(net[4].bias, 0.0)
    return net


class Policy(nn.Module):
    """
    Actor (Policy Network):
    - Input: state vector of 48 floats (or batch of shape [N, 48]).
    - Output: Gaussian distribution over 10 target joint angles.
    - The mean starts at the upright standing pose and can never leave the joint limits, so the
      action that is sampled/evaluated is (almost) the action the simulation actually executes.
    """
    def __init__(self, state_dim=state_dim, action_dim=action_dim):
        super().__init__()
        self.norm = ObsNorm()
        # Near-zero output layer: the untrained policy holds the standing pose
        self.net = _mlp(state_dim, action_dim, out_gain=0.01)

        default = torch.tensor(DEFAULT_POSE, dtype=torch.float32)
        jmin = torch.tensor(JOINT_MIN, dtype=torch.float32)
        jmax = torch.tensor(JOINT_MAX, dtype=torch.float32)
        self.register_buffer("default_pose", default)
        self.register_buffer("joint_min", jmin)
        self.register_buffer("joint_max", jmax)
        self.register_buffer("pos_range", jmax - default)   # room to move up from the default
        self.register_buffer("neg_range", default - jmin)   # room to move down from the default

        # Exploration std dev: exp(-2.0) ~ 0.13 rad (~7.5 degrees)
        self.log_std = nn.Parameter(torch.full((action_dim,), -2.0))

    def mean(self, state):
        t = torch.tanh(self.net(self.norm(state)))
        return self.default_pose + torch.where(t > 0, t * self.pos_range, t * self.neg_range)

    def std(self):
        return self.log_std.clamp(-3.5, -0.5).exp()

    def forward(self, state):
        return Normal(self.mean(state), self.std())

    def sample_action(self, state):
        dist = self.forward(state)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def evaluate(self, state, action):
        dist = self.forward(state)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy

    def deterministic_action(self, state):
        # Best action (mean) with full angular authority to react to impulses
        return self.mean(state)

    def clip_action(self, action):
        """Clamp to the joint limits (what the simulation does anyway), made explicit."""
        return torch.max(torch.min(action, self.joint_max), self.joint_min)


class Critic(nn.Module):
    """
    Critic (Value Network):
    - Predicts expected cumulative future reward V(s) from current state.
    - Used to compute Advantage: A(s, a) = Target_Return - V(s).
    """
    def __init__(self, state_dim=state_dim):
        super().__init__()
        self.norm = ObsNorm()
        self.net = _mlp(state_dim, 1, out_gain=1.0)

    def forward(self, state):
        return self.net(self.norm(state)).squeeze(-1)
