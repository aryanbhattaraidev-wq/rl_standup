"""Reward function."""
import torch


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
