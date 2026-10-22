"""PPO algorithm pieces: generalized advantage estimation and the clipped-surrogate update."""
import torch


def compute_gae(rewards, values, dones, next_value, gamma, gae_lambda):
    """
    rewards/values/dones: lists of T tensors shaped (num_envs,); next_value: (num_envs,) bootstrap.
    `dones` is 1.0 where the body fell on that step (its next state is the respawn, so it is masked out).
    Returns (advantages, returns), both shaped (T, num_envs).
    """
    T = len(rewards)
    advantages = torch.zeros((T, rewards[0].shape[0]), dtype=torch.float32)
    last_gae = torch.zeros(rewards[0].shape[0], dtype=torch.float32)
    for t in reversed(range(T)):
        next_val = next_value if t == T - 1 else values[t + 1]
        non_terminal = 1.0 - dones[t]
        # TD error: delta = r + gamma * V(s') * (1 - done) - V(s)
        delta = rewards[t] + gamma * next_val * non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + torch.stack(values)
    return advantages, returns


def ppo_update(policy, critic, policy_opt, critic_opt, states, actions, old_log_probs, advantages, returns,
               epochs, minibatch, clip_eps, target_kl, entropy_coef):
    """
    Several epochs of minibatch updates with the clipped surrogate objective. Stops early once the
    policy has moved further than target_kl from the one that collected the data.
    All tensor arguments are flat (batch, ...). Returns the mean (actor_loss, critic_loss, kl).
    """
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    batch_size = states.size(0)
    actor_losses, critic_losses, kls = [], [], []
    stop = False
    for _ in range(epochs):
        perm = torch.randperm(batch_size)
        for start in range(0, batch_size, minibatch):
            idx = perm[start:start + minibatch]
            new_log_probs, entropy = policy.evaluate(states[idx], actions[idx])
            new_values = critic(states[idx])

            log_ratio = new_log_probs - old_log_probs[idx]
            ratio = torch.exp(log_ratio)
            adv = advantages[idx]
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
            actor_loss = -torch.min(surr1, surr2).mean() - entropy_coef * entropy.mean()

            policy_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            policy_opt.step()

            critic_loss = 0.5 * ((new_values - returns[idx]) ** 2).mean()
            critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            critic_opt.step()

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

    n = len(kls)
    return sum(actor_losses) / n, sum(critic_losses) / n, sum(kls) / n
