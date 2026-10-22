"""Training entry point: rollouts from the 25-body simulation, PPO updates, logging, checkpoints."""
import argparse
import csv
import os
import time

import torch

from .checkpoint import atomic_save, save_training_state, load_training_state
from .client import Data
from .config import (IP, PY_PORT, SIM_PORT, num_envs, state_dim, action_dim,
                     REWARD_SCALE, FALL_REWARD, CKPT_PATH, BEST_PATH, MODEL_PATH, LOG_PATH)
from .networks import Policy, Critic
from .ppo import compute_gae, ppo_update
from .reward import compute_reward


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


def collect_rollout(udp, policy, critic, states, steps, difficulty, global_step, max_step):
    """
    Runs the policy for up to `steps` simulation steps across all bodies.
    Returns (trajectory dict of per-step tensors, final states, new global_step, number of falls).
    """
    traj = {k: [] for k in ("states", "actions", "log_probs", "rewards", "dones", "values")}
    falls = 0
    for _ in range(steps):
        global_step += 1
        with torch.no_grad():
            action, log_prob = policy.sample_action(states)
            value = critic(states)

        # The unclipped sample is what PPO evaluates; the clipped one is what gets executed
        next_states, dones = udp.step(policy.clip_action(action), (difficulty,))
        rewards = compute_reward(next_states)
        # Fallen bodies get an explicit severe penalty (their next_state is already the respawn)
        rewards = torch.where(dones > 0.5, torch.full_like(rewards, FALL_REWARD), rewards) * REWARD_SCALE

        traj["states"].append(states)
        traj["actions"].append(action)
        traj["log_probs"].append(log_prob)
        traj["rewards"].append(rewards)
        traj["dones"].append(dones)
        traj["values"].append(value)
        falls += int((dones > 0.5).sum().item())

        states = next_states
        if global_step >= max_step:
            break
    return traj, states, global_step, falls


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
    policy_opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.lr * 1.5)

    global_step = 0
    if args.resume and os.path.exists(CKPT_PATH):
        global_step = load_training_state(policy, critic, policy_opt, critic_opt)
        print(f"Resumed from {CKPT_PATH} at step {global_step}")

    gamma, gae_lambda, clip_eps, target_kl = 0.99, 0.95, 0.2, 0.03
    curriculum_steps = max(1, int(args.steps * args.curriculum))
    start_step = global_step

    new_log = not os.path.exists(LOG_PATH) or not args.resume
    log_file = open(LOG_PATH, "a" if args.resume else "w", newline="")
    log = csv.writer(log_file)
    if new_log:
        log.writerow(["step", "difficulty", "falls_per_1k_body_steps", "standing_pct", "mean_reward",
                      "actor_loss", "critic_loss", "kl", "std"])

    best_fall_rate = float("inf")
    t0 = time.time()
    print(f"\nStarting 25-Body PPO Training ({args.rollout * num_envs} transitions per update, "
          f"{args.steps} steps total)...")
    print("Press Ctrl+C at any time to pause and save weights.\n")

    try:
        while global_step < args.steps:
            # Linear learning-rate decay and push-difficulty curriculum
            frac = 1.0 - global_step / args.steps
            for opt, base in ((policy_opt, args.lr), (critic_opt, args.lr * 1.5)):
                for g in opt.param_groups:
                    g["lr"] = max(base * frac, base * 0.1)
            difficulty = min(1.0, 0.2 + 0.8 * global_step / curriculum_steps)

            # 1. Rollout collection (vectorized across 25 bodies)
            traj, states, global_step, falls = collect_rollout(
                udp, policy, critic, states, args.rollout, difficulty, global_step, args.steps)

            # 2. Generalized Advantage Estimation
            with torch.no_grad():
                next_value = critic(states)
            advantages, returns = compute_gae(traj["rewards"], traj["values"], traj["dones"],
                                              next_value, gamma, gae_lambda)

            # 3. PPO update on the flattened batch
            a_loss, c_loss, mean_kl = ppo_update(
                policy, critic, policy_opt, critic_opt,
                torch.stack(traj["states"]).view(-1, state_dim),
                torch.stack(traj["actions"]).view(-1, action_dim),
                torch.stack(traj["log_probs"]).view(-1),
                advantages.view(-1), returns.view(-1),
                epochs=args.epochs, minibatch=args.minibatch, clip_eps=clip_eps,
                target_kl=target_kl, entropy_coef=args.entropy)

            # 4. Logging + checkpoints (averaged over the whole update, not just the last minibatch)
            n_trans = len(traj["rewards"]) * num_envs
            fall_rate = 1000.0 * falls / n_trans
            standing_pct = 100.0 * ((states[:, 34] >= 120.0) & (states[:, 28] >= 0.5)).float().mean().item()
            mean_reward = torch.stack(traj["rewards"]).mean().item() / REWARD_SCALE
            std = policy.std().mean().item()
            sps = (global_step - start_step) / max(1e-6, time.time() - t0)
            print(f"Step {global_step:7d} | diff {difficulty:.2f} | falls/1k {fall_rate:6.1f} | standing {standing_pct:5.1f}% | "
                  f"reward {mean_reward:6.2f} | actor {a_loss:7.4f} critic {c_loss:7.3f} kl {mean_kl:.4f} std {std:.3f} | {sps:5.0f} step/s")
            log.writerow([global_step, f"{difficulty:.3f}", f"{fall_rate:.2f}", f"{standing_pct:.1f}",
                          f"{mean_reward:.3f}", f"{a_loss:.5f}", f"{c_loss:.4f}", f"{mean_kl:.5f}", f"{std:.4f}"])
            log_file.flush()

            save_training_state(policy, critic, policy_opt, critic_opt, global_step)
            # "Best" only counts once pushes are at full difficulty so it is comparable across updates
            if difficulty >= 0.999 and fall_rate < best_fall_rate:
                best_fall_rate = fall_rate
                atomic_save(policy.state_dict(), BEST_PATH)

    except KeyboardInterrupt:
        print("\nTraining paused by user.")
    finally:
        save_training_state(policy, critic, policy_opt, critic_opt, global_step)
        log_file.close()
        print(f"[Saved] Model checkpoint saved to {MODEL_PATH}"
              + (f" (best at full difficulty: {BEST_PATH}, {best_fall_rate:.1f} falls/1k)" if best_fall_rate < float('inf') else ""))


if __name__ == "__main__":
    main()
