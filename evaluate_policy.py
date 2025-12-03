import torch
import numpy as np
import matplotlib.pyplot as plt
from chemo_env import ChemotherapyEnv
from ppo_train import MLPActorCritic


def evaluate(num_steps=500):
    env = ChemotherapyEnv()
    obs_dim = 6

    # Load model
    model = MLPActorCritic(obs_dim)
    model.load_state_dict(torch.load("ppo_chemotherapy.pt", map_location="cpu"))
    model.eval()

    # Reset environment
    obs = env.reset()
    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)

    # Storage for plotting
    states = []
    actions = []
    rewards = []

    for _ in range(num_steps):
        with torch.no_grad():
            action, logp, value = model.act(obs_t)
        action_scalar = float(action.numpy()[0, 0])

        next_obs, reward, done, info = env.step(action_scalar)

        states.append(next_obs)
        actions.append(action_scalar)
        rewards.append(reward)

        obs_t = torch.as_tensor(next_obs, dtype=torch.float32).unsqueeze(0)

        if done:
            break

    states = np.array(states)
    actions = np.array(actions)

    # Unpack states
    Pc, Qc, Cc = states[:,0], states[:,1], states[:,2]
    Pb, Qb, Cb = states[:,3], states[:,4], states[:,5]

    t = np.arange(len(states)) * env.dt

    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    axes[0].plot(t, Pc, label="Proliferative Cancer (P)")
    axes[0].plot(t, Qc, label="Quiescent Cancer (Q)")
    axes[0].plot(t, Cc, label="Cancer Concentration")
    axes[0].set_title("Cancer Cell Dynamics")
    axes[0].set_xlabel("Time (days)")
    axes[0].legend()

    axes[1].plot(t, Pb, label="Proliferative Bone Marrow (P)")
    axes[1].plot(t, Qb, label="Quiescent Bone Marrow (Q)")
    axes[1].plot(t, Cb, label="Bone Marrow Concentration")
    axes[1].set_title("Bone Marrow Dynamics")
    axes[1].set_xlabel("Time (days)")
    axes[1].legend()

    axes[2].plot(t, actions, label="PPO Dosing u(t)")
    axes[2].set_title("PPO Learned Drug Infusion Schedule")
    axes[2].set_xlabel("Time (days)")
    axes[2].set_ylabel("Infusion Rate")
    axes[2].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    evaluate()
