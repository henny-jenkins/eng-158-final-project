# ppo_train.py

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from chemo_env import ChemotherapyEnv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden_sizes=(64, 64)):
        super().__init__()

        layers = []
        last_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.Tanh())
            last_dim = h
        self.shared = nn.Sequential(*layers)

        self.mu_head = nn.Linear(last_dim, 1)        # scalar action
        self.log_std = nn.Parameter(torch.zeros(1))  # global log-std

        self.v_head = nn.Linear(last_dim, 1)

    def forward(self, obs: torch.Tensor):
        x = self.shared(obs)
        mu = self.mu_head(x)
        std = torch.exp(self.log_std)
        v = self.v_head(x)
        return mu, std, v.squeeze(-1)

    def act(self, obs: torch.Tensor):
        mu, std, v = self.forward(obs)
        dist = Normal(mu, std)
        raw_action = dist.rsample()

        # Tanh squash to [-1, 1] and map to [0, 1]
        action_tanh = torch.tanh(raw_action)
        action = 0.5 * (action_tanh + 1.0)

        log_prob = dist.log_prob(raw_action).sum(-1)

        return action.detach(), log_prob.detach(), v.detach()


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    Generalized Advantage Estimation (GAE).
    values: length T+1 (bootstrapped last value)
    rewards, dones: length T
    """
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_nonterminal = 1.0 - dones[t]
        next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        adv[t] = last_gae
    returns = adv + values[:-1]
    return adv, returns


def ppo_train(
    total_steps=100_000,
    update_interval=2048,
    minibatch_size=256,
    epochs=10,
    gamma=0.99,
    lam=0.95,
    clip_ratio=0.2,
    pi_lr=3e-4,
    v_coef=0.5,
    ent_coef=0.0,
):

    env = ChemotherapyEnv()
    obs_dim = 6  # [Pc, Qc, Cc, Pb, Qb, Cb]

    actor_critic = MLPActorCritic(obs_dim).to(device)
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=pi_lr)

    obs = env.reset()
    obs = torch.as_tensor(obs, dtype=torch.float32, device=device)

    step_count = 0

    while step_count < total_steps:
        # Storage for one rollout
        obs_buf = []
        act_buf = []
        logp_buf = []
        rew_buf = []
        val_buf = []
        done_buf = []

        # Collect experience
        for _ in range(update_interval):
            with torch.no_grad():
                action, logp, value = actor_critic.act(obs.unsqueeze(0))
            action_scalar = float(action.cpu().numpy()[0, 0])

            next_obs, reward, done, info = env.step(action_scalar)

            obs_buf.append(obs.cpu().numpy())
            act_buf.append(action_scalar)
            logp_buf.append(logp.cpu().numpy())
            rew_buf.append(reward)
            val_buf.append(value.cpu().numpy())
            done_buf.append(float(done))

            obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)

            step_count += 1
            if done:
                obs = torch.as_tensor(env.reset(), dtype=torch.float32, device=device)

            if step_count >= total_steps:
                break

        # Bootstrap value for final state
        with torch.no_grad():
            _, _, last_value = actor_critic.act(obs.unsqueeze(0))
        val_buf.append(last_value.cpu().numpy())

        # Convert to arrays
        obs_buf = np.array(obs_buf, dtype=np.float32)
        act_buf = np.array(act_buf, dtype=np.float32)
        logp_buf = np.array(logp_buf, dtype=np.float32).reshape(-1)
        rew_buf = np.array(rew_buf, dtype=np.float32)
        val_buf = np.array(val_buf, dtype=np.float32).reshape(-1)
        done_buf = np.array(done_buf, dtype=np.float32)

        # Compute advantages and returns
        adv_buf, ret_buf = compute_gae(
            rewards=rew_buf,
            values=val_buf,
            dones=done_buf,
            gamma=gamma,
            lam=lam,
        )

        # Normalize advantages
        adv_mean, adv_std = adv_buf.mean(), adv_buf.std() + 1e-8
        adv_buf = (adv_buf - adv_mean) / adv_std

        # Convert all to tensors
        obs_t = torch.as_tensor(obs_buf, dtype=torch.float32, device=device)
        act_t = torch.as_tensor(act_buf, dtype=torch.float32, device=device).unsqueeze(-1)
        logp_t = torch.as_tensor(logp_buf, dtype=torch.float32, device=device)
        adv_t = torch.as_tensor(adv_buf, dtype=torch.float32, device=device)
        ret_t = torch.as_tensor(ret_buf, dtype=torch.float32, device=device)

        dataset_size = len(obs_buf)

        # PPO updates
        for _ in range(epochs):
            idxs = np.arange(dataset_size)
            np.random.shuffle(idxs)
            for start in range(0, dataset_size, minibatch_size):
                end = start + minibatch_size
                batch_idx = idxs[start:end]

                b_obs = obs_t[batch_idx]
                b_act = act_t[batch_idx]
                b_logp_old = logp_t[batch_idx]
                b_adv = adv_t[batch_idx]
                b_ret = ret_t[batch_idx]

                # Forward pass
                mu, std, v_pred = actor_critic(b_obs)
                dist = Normal(mu, std)

                # Approximate inverse of squash: a in [0,1] -> raw in [-1,1]
                raw_action = (b_act * 2.0 - 1.0).clamp(-0.999, 0.999)
                logp = dist.log_prob(raw_action).sum(-1)
                entropy = dist.entropy().sum(-1)

                # Policy loss (clipped surrogate)
                ratio = torch.exp(logp - b_logp_old)
                unclipped = ratio * b_adv
                clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * b_adv
                pi_loss = -torch.min(unclipped, clipped).mean()

                # Value loss
                v_loss = ((v_pred - b_ret) ** 2).mean()

                # Total loss (single backward pass)
                loss = pi_loss + v_coef * v_loss - ent_coef * entropy.mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor_critic.parameters(), 0.5)
                optimizer.step()

        avg_ret = ret_buf.mean()
        avg_rew = rew_buf.mean()
        print(f"Steps: {step_count:7d} | Avg return: {avg_ret:8.3f} | Avg reward: {avg_rew:8.3f}")

    torch.save(actor_critic.state_dict(), "ppo_chemotherapy.pt")
    print("Training complete. Model saved to ppo_chemotherapy.pt")


if __name__ == "__main__":
    ppo_train()
