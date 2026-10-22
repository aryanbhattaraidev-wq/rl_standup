"""Constants shared by the whole agent. Must match the C++ side (sim/Protocol.h, sim/physics/Skeleton.h)."""
import os

# --- UDP ---------------------------------------------------------------------------------------
IP = "127.0.0.1"
SIM_PORT = int(os.environ.get("STANDUP_SIM_PORT", 5005))  # C++ listens here
PY_PORT = int(os.environ.get("STANDUP_PY_PORT", 5006))    # Python listens here

# --- Dimensions --------------------------------------------------------------------------------
num_envs = 25      # parallel ragdolls in training mode
state_dim = 48     # 11 bones * 3 (cos, sin, angVel*0.01) + 4 hip + 4 footL + 4 footR + 3 chest
action_dim = 10    # 10 joints per body
packet_dim = state_dim + 1  # state + done flag per body

# Joint limits [min, max] and the upright standing pose, in joint order:
# 0: Neck, 1: Waist, 2: ShoulderL, 3: ShoulderR, 4: ElbowL, 5: ElbowR, 6: HipL, 7: HipR, 8: KneeL, 9: KneeR
JOINT_MIN = [-0.8, -2.5, -2.8, -3.2, -0.3, -2.6, 0.4, -2.4, -0.2, 1.2]
JOINT_MAX = [0.8, -0.6, 1.2, -0.8, 2.6, 0.3, 2.4, -0.4, 2.5, 3.3]
DEFAULT_POSE = [0.0, -1.57, -0.90, -2.24, 0.38, -0.38, 1.37, -1.37, 0.21, 2.94]

# --- Rewards -----------------------------------------------------------------------------------
# Rewards are divided by REWARD_SCALE so value targets stay O(10-100) instead of O(1000)
REWARD_SCALE = 0.1
FALL_REWARD = -10.0

# --- Output files (relative to the working directory) ------------------------------------------
MODEL_PATH = "standup_policy.pth"       # latest policy weights (what test.py loads)
BEST_PATH = "standup_policy_best.pth"   # best policy at full push difficulty
CKPT_PATH = "standup_ckpt.pth"          # policy + critic + optimizers, for --resume
LOG_PATH = "training_log.csv"
