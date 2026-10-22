import argparse
import csv
import os
import socket
import struct
import time

import torch
import torch.nn as nn
from torch.distributions import Normal

# --- Protocol / environment constants (must match rl_standup.cpp and Skeleton.h) -------------------
IP = "127.0.0.1"
SIM_PORT = int(os.environ.get("STANDUP_SIM_PORT", 5005))  # C++ listens here
PY_PORT = int(os.environ.get("STANDUP_PY_PORT", 5006))    # Python listens here

num_envs = 25      # parallel ragdolls in training mode
state_dim = 48     # 11 bones * 3 (cos, sin, angVel*0.01) + 4 hip + 4 footL + 4 footR + 3 chest
action_dim = 10    # 10 joints per body
packet_dim = state_dim + 1  # state + done flag per body

# Joint limits [min, max] and the upright standing pose, in joint order:
# 0: Neck, 1: Waist, 2: ShoulderL, 3: ShoulderR, 4: ElbowL, 5: ElbowR, 6: HipL, 7: HipR, 8: KneeL, 9: KneeR
JOINT_MIN = [-0.8, -2.5, -2.8, -3.2, -0.3, -2.6, 0.4, -2.4, -0.2, 1.2]
JOINT_MAX = [0.8, -0.6, 1.2, -0.8, 2.6, 0.3, 2.4, -0.4, 2.5, 3.3]
DEFAULT_POSE = [0.0, -1.57, -0.90, -2.24, 0.38, -0.38, 1.37, -1.37, 0.21, 2.94]

# Rewards are divided by this so value targets stay O(10-100) instead of O(1000)
REWARD_SCALE = 0.1
FALL_REWARD = -10.0

MODEL_PATH = "standup_policy.pth"
BEST_PATH = "standup_policy_best.pth"
CKPT_PATH = "standup_ckpt.pth"
LOG_PATH = "training_log.csv"


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

    def forward(self, state):
        std = self.log_std.clamp(-3.5, -0.5).exp()
        return Normal(self.mean(state), std)

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


class Data:
    """
    UDP client for the C++ simulation (lockstep: one action packet in, one state packet out).
    n_envs = 25 with handshake -100 for training, n_envs = 1 with handshake -101 for solo testing.
    """
    def __init__(self, ip=IP, recv_port=PY_PORT, send_port=SIM_PORT, n_envs=num_envs,
                 handshake=-100.0, timeout=10.0):
        self.n_envs = n_envs
        self.handshake = handshake
        self.recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.recv_sock.bind((ip, recv_port))
        # Fail loudly instead of hanging forever if the simulation is not running
        self.recv_sock.settimeout(timeout)

        self.send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.send_addr = (ip, send_port)

    def receive_state(self):
        expected = self.n_envs * packet_dim
        try:
            data, _ = self.recv_sock.recvfrom(65536)
        except socket.timeout:
            raise RuntimeError("Timed out waiting for the simulation. Is rl_standup.exe running?")
        num_floats = len(data) // 4
        if num_floats != expected:
            raise RuntimeError(
                f"\n[Mode Mismatch] Received {num_floats} floats but expected {expected} "
                f"({self.n_envs} bodies).\n"
                "--> Restart rl_standup.exe (it keeps the mode of whichever script connected first)."
            )
        raw = torch.frombuffer(bytearray(data), dtype=torch.float32)
        return raw.view(self.n_envs, packet_dim).clone()

    def send(self, floats):
        self.send_sock.sendto(struct.pack(f"{len(floats)}f", *floats), self.send_addr)

    def drain(self):
        """Drop stale packets left over from a previous run so states and actions stay in lockstep."""
        self.recv_sock.settimeout(0.0)
        try:
            while True:
                self.recv_sock.recvfrom(65536)
        except (BlockingIOError, socket.timeout):
            pass
        finally:
            self.recv_sock.settimeout(10.0)

    def get_state(self, extra=()):
        # Handshake: [h, h, extra...] resets all bodies and requests their initial state
        self.drain()
        self.send([self.handshake, self.handshake, *extra])
        return self._split(self.receive_state())

    def step(self, actions_tensor, extra=()):
        # actions_tensor: shape (n_envs, action_dim); `extra` floats are appended to the packet
        flat = actions_tensor.contiguous().view(-1).tolist()
        self.send(flat + list(extra))
        return self._split(self.receive_state())

    @staticmethod
    def _split(raw):
        return raw[:, :state_dim], raw[:, state_dim]


def compute_reward(state, target_height=160.0):
    """
    Vectorized reward function supporting both (48,) and (num_envs, 48):
    1. Hip Height: scaled as hip approaches standing height (160 px).
    2. Chest Upright: Chest (Bone 9) must point straight UP (sin_chest -> 1.0).
    3. Head Upright: Head (Bone 10) pointing straight UP (sin_head -> 1.0).
    4. Level Hip: Pelvic crossbar stays horizontal (sin_hip -> 0.0).
    5. Foot Stance Width: Natural base of support (~50-80 px), no leg splits or crossing.
    6. Ground Contact: Feet stay near floor rather than flailing in the air.
    7. Balance Alignment: Hip stays centered between feet base-of-support.
    8. Posture Penalty: Heavy penalty if folded downward (V-shape) or hip collapsed.
    """
    single = (state.dim() == 1)
    if single:
        state = state.unsqueeze(0)

    # Hip (Bone 8)
    sin_hip  = state[:, 25]
    hip_relX = state[:, 33]
    hip_y    = state[:, 34]
    hip_vx   = state[:, 35]
    hip_vy   = state[:, 36]

    # Chest (Bone 9) & Head (Bone 10)
    sin_chest = state[:, 28]  # Vertical spine: sin -> 1.0 when upright
    sin_head  = state[:, 31]  # Head: sin -> 1.0 when upright

    # Feet endpoints: Left (indices 37-40), Right (indices 41-44)
    footL_relX = state[:, 37]
    footL_y    = state[:, 38]
    footR_relX = state[:, 41]
    footR_y    = state[:, 42]

    # 1. Hip Height: rewards hip near 160 px
    height_ratio = torch.clamp((hip_y - 70.0) / (target_height - 70.0), 0.0, 1.2)
    hip_height_reward = 3.0 * height_ratio

    # 2. Chest & Head Upright: heavily rewards chest pointing straight up
    chest_reward = 3.5 * torch.clamp(sin_chest, -1.0, 1.0)
    head_reward  = 1.0 * torch.clamp(sin_head, -1.0, 1.0)

    # 3. Level Hip: pelvic crossbar stays horizontal
    flat_hip_reward = 1.5 * (1.0 - (sin_hip ** 2))

    # 4. Foot Stance Width: penalizes outward splits (>85 px) and crossing legs (<35 px)
    stance_width = torch.abs(footR_relX - footL_relX)
    split_penalty = -3.0 * torch.clamp((stance_width - 85.0) / 30.0, min=0.0)**2
    cross_penalty = -2.0 * torch.clamp((35.0 - stance_width) / 20.0, min=0.0)**2
    stance_penalty = split_penalty + cross_penalty

    # 5. Ground Contact: rewards keeping feet on/near the floor (floor at y = 20)
    ground_contact_reward = 0.5 * (
        torch.clamp(1.0 - (footL_y - 20.0) / 40.0, 0.0, 1.0) +
        torch.clamp(1.0 - (footR_y - 20.0) / 40.0, 0.0, 1.0)
    )

    # 6. Balance Alignment: keeps hip centered over base of support (midpoint of feet)
    foot_midpoint = (footL_relX + footR_relX) * 0.5
    com_offset = torch.abs(hip_relX - foot_midpoint)
    balance_alignment_penalty = -1.5 * torch.clamp((com_offset - 25.0) / 30.0, min=0.0)**2

    # 7. Stability: negligible velocity penalty so AI can react rapidly without fear
    stability_penalty = -0.00002 * (hip_vx**2 + hip_vy**2)

    # 8. Severe penalty if folded into a V-shape (sin_chest < 0.0) or hip < 70
    is_bad_posture = (sin_chest < 0.0) | (hip_y < 70.0)
    posture_reward = torch.where(is_bad_posture, -8.0, 2.0)

    total_reward = (
        hip_height_reward
        + chest_reward
        + head_reward
        + flat_hip_reward
        + stance_penalty
        + ground_contact_reward
        + balance_alignment_penalty
        + stability_penalty
        + posture_reward
    )
    return total_reward.squeeze(0) if single else total_reward


def atomic_save(obj, path):
    """Write to a temp file first so a crash mid-save cannot corrupt an existing checkpoint."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def parse_args():
    p = argparse.ArgumentParser(description="PPO training of the standing ragdoll (needs rl_standup.exe running)")
    p.add_argument("--steps", type=int, default=300_000, help="total simulation steps (each steps all 25 bodies)")
    p.add_argument("--rollout", type=int, default=256, help="steps collected per PPO update")
    p.add_argument("--epochs", type=int, default=8, help="PPO epochs per update")
    p.add_argument("--minibatch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4, help="actor learning rate (critic uses 1.5x)")
    p.add_argument("--curriculum", type=float, default=0.4,
                   help="fraction of training over which push difficulty ramps from 0.2 to 1.0")
    p.add_argument("--init-log-std", type=float, default=-2.0, help="initial exploration noise (log of std, radians)")
    p.add_argument("--entropy", type=float, default=0.002, help="entropy bonus coefficient")
    p.add_argument("--threads", type=int, default=4, help="torch CPU threads")
    p.add_argument("--resume", action="store_true", help=f"continue from {CKPT_PATH}")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    udp = Data(IP, PY_PORT, SIM_PORT, num_envs, handshake=-100.0)
    print(f"Connecting to 25-body C++ simulation on {IP}:{SIM_PORT}...")
    difficulty = 0.2
    states, dones = udp.get_state((difficulty,))
    print(f"Connected! Initial states shape: {tuple(states.shape)} (25 bodies parallel)")

    torch.set_num_threads(args.threads)
    policy = Policy()
    with torch.no_grad():
        policy.log_std.fill_(args.init_log_std)
    critic = Critic()
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.lr * 1.5)

    global_step = 0
    if args.resume and os.path.exists(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
        policy.load_state_dict(ckpt["policy"])
        critic.load_state_dict(ckpt["critic"])
        policy_optimizer.load_state_dict(ckpt["policy_opt"])
        critic_optimizer.load_state_dict(ckpt["critic_opt"])
        global_step = int(ckpt["step"])
        print(f"Resumed from {CKPT_PATH} at step {global_step}")

    gamma, gae_lambda, clip_eps, target_kl = 0.99, 0.95, 0.2, 0.03
    entropy_coef = args.entropy
    curriculum_steps = max(1, int(args.steps * args.curriculum))
    start_step = global_step

    new_log = not os.path.exists(LOG_PATH) or not args.resume
    log_file = open(LOG_PATH, "a" if args.resume else "w", newline="")
    log = csv.writer(log_file)
    if new_log:
        log.writerow(["step", "difficulty", "falls_per_1k_body_steps", "standing_pct", "mean_reward",
                      "actor_loss", "critic_loss", "kl", "std"])

    def save_all(path_policy=MODEL_PATH):
        atomic_save(policy.state_dict(), path_policy)
        atomic_save({"policy": policy.state_dict(), "critic": critic.state_dict(),
                     "policy_opt": policy_optimizer.state_dict(), "critic_opt": critic_optimizer.state_dict(),
                     "step": global_step}, CKPT_PATH)

    best_fall_rate = float("inf")
    t0 = time.time()
    print(f"\nStarting 25-Body PPO Training ({args.rollout * num_envs} transitions per update, "
          f"{args.steps} steps total)...")
    print("Press Ctrl+C at any time to pause and save weights.\n")

    try:
        while global_step < args.steps:
            # Linear learning-rate decay and push-difficulty curriculum
            frac = 1.0 - global_step / args.steps
            for opt, base in ((policy_optimizer, args.lr), (critic_optimizer, args.lr * 1.5)):
                for g in opt.param_groups:
                    g["lr"] = max(base * frac, base * 0.1)
            difficulty = min(1.0, 0.2 + 0.8 * global_step / curriculum_steps)

            b_states, b_actions, b_log_probs = [], [], []
            b_rewards, b_dones, b_values = [], [], []
            falls = 0

            # 1. Rollout collection (vectorized across 25 bodies)
            for _ in range(args.rollout):
                global_step += 1
                with torch.no_grad():
                    action, log_prob = policy.sample_action(states)
                    value = critic(states)

                # The unclipped sample is what PPO evaluates; the clipped one is what gets executed
                next_states, dones = udp.step(policy.clip_action(action), (difficulty,))
                rewards = compute_reward(next_states)
                # Fallen bodies get an explicit severe penalty (their next_state is already the respawn)
                rewards = torch.where(dones > 0.5, torch.full_like(rewards, FALL_REWARD), rewards) * REWARD_SCALE

                b_states.append(states)
                b_actions.append(action)
                b_log_probs.append(log_prob)
                b_rewards.append(rewards)
                b_dones.append(dones)
                b_values.append(value)
                falls += int((dones > 0.5).sum().item())

                states = next_states
                if global_step >= args.steps:
                    break

            # 2. Generalized Advantage Estimation (GAE)
            with torch.no_grad():
                next_value = critic(states)

            T = len(b_rewards)
            b_advantages = torch.zeros((T, num_envs), dtype=torch.float32)
            last_gae = torch.zeros(num_envs, dtype=torch.float32)
            for t in reversed(range(T)):
                next_val = next_value if t == T - 1 else b_values[t + 1]
                non_terminal = 1.0 - b_dones[t]
                # TD error: delta = r + gamma * V(s') * (1 - done) - V(s)
                delta = b_rewards[t] + gamma * next_val * non_terminal - b_values[t]
                last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
                b_advantages[t] = last_gae
            b_returns = b_advantages + torch.stack(b_values)

            flat_states = torch.stack(b_states).view(-1, state_dim)
            flat_actions = torch.stack(b_actions).view(-1, action_dim)
            flat_log_probs = torch.stack(b_log_probs).view(-1)
            flat_advantages = b_advantages.view(-1)
            flat_returns = b_returns.view(-1)
            flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

            # 3. PPO epochs with clipped surrogate objective; stop early if the policy moves too far
            batch_size = flat_states.size(0)
            actor_losses, critic_losses, kls = [], [], []
            stop = False
            for epoch in range(args.epochs):
                perm = torch.randperm(batch_size)
                for start in range(0, batch_size, args.minibatch):
                    idx = perm[start:start + args.minibatch]
                    new_log_probs, entropy = policy.evaluate(flat_states[idx], flat_actions[idx])
                    new_values = critic(flat_states[idx])

                    log_ratio = new_log_probs - flat_log_probs[idx]
                    ratio = torch.exp(log_ratio)
                    adv = flat_advantages[idx]
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
                    actor_loss = -torch.min(surr1, surr2).mean() - entropy_coef * entropy.mean()

                    policy_optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                    policy_optimizer.step()

                    critic_loss = 0.5 * ((new_values - flat_returns[idx]) ** 2).mean()
                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                    critic_optimizer.step()

                    with torch.no_grad():
                        kl = ((ratio - 1.0) - log_ratio).mean().item()
                    actor_losses.append(actor_loss.item())
                    critic_losses.append(critic_loss.item())
                    kls.append(kl)
                    if kl > target_kl:
                        stop = True
                        break
                if stop:
                    break

            # 4. Logging + checkpoints (averaged over the whole update, not just the last minibatch)
            n_trans = T * num_envs
            fall_rate = 1000.0 * falls / n_trans
            standing_pct = 100.0 * ((states[:, 34] >= 120.0) & (states[:, 28] >= 0.5)).float().mean().item()
            mean_reward = torch.stack(b_rewards).mean().item() / REWARD_SCALE
            a_loss = sum(actor_losses) / len(actor_losses)
            c_loss = sum(critic_losses) / len(critic_losses)
            mean_kl = sum(kls) / len(kls)
            std = policy.log_std.clamp(-3.5, -0.5).exp().mean().item()
            sps = (global_step - start_step) / max(1e-6, time.time() - t0)
            print(f"Step {global_step:7d} | diff {difficulty:.2f} | falls/1k {fall_rate:6.1f} | standing {standing_pct:5.1f}% | "
                  f"reward {mean_reward:6.2f} | actor {a_loss:7.4f} critic {c_loss:7.3f} kl {mean_kl:.4f} std {std:.3f} | {sps:5.0f} step/s")
            log.writerow([global_step, f"{difficulty:.3f}", f"{fall_rate:.2f}", f"{standing_pct:.1f}",
                          f"{mean_reward:.3f}", f"{a_loss:.5f}", f"{c_loss:.4f}", f"{mean_kl:.5f}", f"{std:.4f}"])
            log_file.flush()

            save_all()
            # "Best" only counts once pushes are at full difficulty so it is comparable across updates
            if difficulty >= 0.999 and fall_rate < best_fall_rate:
                best_fall_rate = fall_rate
                atomic_save(policy.state_dict(), BEST_PATH)

    except KeyboardInterrupt:
        print("\nTraining paused by user.")
    finally:
        save_all()
        log_file.close()
        print(f"[Saved] Model checkpoint saved to {MODEL_PATH}"
              + (f" (best at full difficulty: {BEST_PATH}, {best_fall_rate:.1f} falls/1k)" if best_fall_rate < float('inf') else ""))


if __name__ == "__main__":
    main()
