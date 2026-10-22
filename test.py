import argparse
import os
import time

import torch

from train import Policy, Data, compute_reward, MODEL_PATH, BEST_PATH, IP, PY_PORT, SIM_PORT


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate the trained standing policy (needs rl_standup.exe running)")
    p.add_argument("--baseline", action="store_true", help="ignore weights, hold the default standing pose")
    p.add_argument("--model", default=None, help=f"checkpoint path (default {MODEL_PATH})")
    p.add_argument("--best", action="store_true", help=f"use {BEST_PATH} (best fall rate seen during training)")
    p.add_argument("--stress", type=float, default=0.0, metavar="D",
                   help="automatic shoves + flying balls at difficulty D in (0, 1]; 0 = only your own mouse")
    p.add_argument("--duration", type=float, default=None, help="stop after this many simulated seconds")
    p.add_argument("--fast", action="store_true", help="don't sleep to 60 FPS (benchmarking)")
    return p.parse_args()


def load_policy(args):
    policy = Policy()
    path = args.model or (BEST_PATH if args.best else MODEL_PATH)
    if args.baseline:
        print("\n[Baseline Mode] Holding the default standing pose (no learned weights).")
    elif not os.path.exists(path):
        print(f"\n[Notice] '{path}' not found. Holding the default standing pose.")
    else:
        print(f"\nLoading trained model from '{path}'...")
        try:
            policy.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            print("Successfully loaded trained weights.")
        except (RuntimeError, OSError) as e:
            raise SystemExit(f"[Error] Could not load '{path}': {e}\n"
                             "It was probably trained with an older version of train.py - retrain, or use --baseline.")
    policy.eval()
    return policy


def run_evaluation():
    args = parse_args()
    policy = load_policy(args)
    print("Policy ready in Evaluation Mode (Pure Learned Reflexes, Zero Noise).")

    udp = Data(IP, PY_PORT, SIM_PORT, n_envs=1, handshake=-101.0)
    print("\nConnecting to C++ simulation (Requesting 1-Body Solo Mode)...")
    state, done = udp.get_state((args.stress,) if args.stress > 0 else ())
    print("Connected! Initial state received for single centered ragdoll.\n")

    print("=" * 75)
    print("  STANDUP AI - SOLO MODEL EVALUATION (1 RAGDOLL)")
    print("  - Target Standing Hip Height : 160.0 px (green guideline)")
    print("  - Random Pose Spawning       : Arms & limbs spawn in randomized poses")
    if args.stress > 0:
        print(f"  - STRESS TEST                : automatic shoves + red balls, difficulty {args.stress:.2f}")
    print("  - Controls in OpenGL Window  :")
    print("      * Press 'R'              : Spawn a fresh randomized test pose on demand")
    print("      * Press 'SPACE'          : Deliver a sharp impulse shove to test balance")
    print("      * Glowing Orange Ball    : Move mouse to push & deflect ragdoll limbs")
    print("      * Left Click & Drag      : Drag any individual limb directly")
    print("  - Stop                       : Press Ctrl + C in this terminal")
    print("=" * 75)

    step = 0
    total_reward = 0.0
    start_time = time.time()
    total_falls = 0
    current_streak = 0
    max_streak = 0
    target_frame_time = 1.0 / 60.0
    max_steps = int(args.duration * 60) if args.duration else None

    try:
        with torch.no_grad():
            while max_steps is None or step < max_steps:
                frame_start = time.perf_counter()
                step += 1

                # A. Deterministic policy action (mean of distribution, no exploration noise)
                action = policy.clip_action(policy.deterministic_action(state))

                # B. Step physics simulation in C++
                state, done = udp.step(action)

                # C. Step reward
                total_reward += compute_reward(state).item()

                # D. Fall & balance statistics
                is_done = bool(done.item() > 0.5)
                hip_height = state[0, 34].item()

                if is_done:
                    total_falls += 1
                    current_streak = 0
                    status = "FALLEN/RESET"
                else:
                    current_streak += 1
                    max_streak = max(max_streak, current_streak)

                    sin_chest = state[0, 28].item()
                    if sin_chest < 0.3:
                        status = "FOLDED V    "
                    elif hip_height >= 140.0 and sin_chest >= 0.7:
                        status = "STAND TALL  "
                    elif hip_height >= 110.0:
                        status = "CROUCHING   "
                    else:
                        status = "RECOVERING  "

                # Live telemetry 5 times per second of simulated time
                if step % 12 == 0 and not args.fast:
                    print(
                        f"Time: {step / 60.0:5.1f}s | [{status}] Hip Height: {hip_height:5.1f}px (Target: 160px) | "
                        f"Standing Streak: {current_streak / 60.0:5.1f}s (Best: {max_streak / 60.0:5.1f}s) | "
                        f"Falls: {total_falls:2d}"
                    )

                # Maintain 60 FPS so the animation is smooth and real-time
                if not args.fast:
                    sleep_time = target_frame_time - (time.perf_counter() - frame_start)
                    if sleep_time > 0:
                        time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass

    elapsed = time.time() - start_time
    sim_minutes = step / 60.0 / 60.0
    print("\n" + "=" * 75)
    print("  EVALUATION SUMMARY")
    print("=" * 75)
    print(f"  Total Steps Evaluated  : {step:,} ({step / 60.0:.1f}s simulated, {step / max(elapsed, 1e-6):.0f} steps/sec)")
    print(f"  Total Falls Recorded   : {total_falls} ({total_falls / max(sim_minutes, 1e-9):.2f} per simulated minute)")
    print(f"  Longest Balance Streak : {max_streak / 60.0:.2f} seconds ({max_streak} steps)")
    print(f"  Average Step Reward    : {total_reward / max(1, step):.3f}")
    print("=" * 75)


if __name__ == "__main__":
    run_evaluation()
