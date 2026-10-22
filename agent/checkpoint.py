"""Saving and loading training state."""
import os

import torch

from .config import MODEL_PATH, CKPT_PATH


def atomic_save(obj, path):
    """Write to a temp file first so a crash mid-save cannot corrupt an existing checkpoint."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_training_state(policy, critic, policy_opt, critic_opt, step):
    """Latest policy weights (for test.py) plus everything needed to --resume."""
    atomic_save(policy.state_dict(), MODEL_PATH)
    atomic_save({"policy": policy.state_dict(), "critic": critic.state_dict(),
                 "policy_opt": policy_opt.state_dict(), "critic_opt": critic_opt.state_dict(),
                 "step": step}, CKPT_PATH)


def load_training_state(policy, critic, policy_opt, critic_opt):
    """Restores state saved by save_training_state and returns the step it was saved at."""
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    policy.load_state_dict(ckpt["policy"])
    critic.load_state_dict(ckpt["critic"])
    policy_opt.load_state_dict(ckpt["policy_opt"])
    critic_opt.load_state_dict(ckpt["critic_opt"])
    return int(ckpt["step"])
