# chemo_mpc_plot.py

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.optimize import minimize
from chemo_env import ChemotherapyEnv

def simulate_horizon(u_seq, env, state):
    """
    Simulate environment over horizon using candidate control sequence.
    Returns negative total reward (for minimization).
    """
    sim_env = deepcopy(env)
    sim_env.state = state.copy()
    sim_env.t = env.t
    total_reward = 0.0

    for u in u_seq:
        _, reward, done, _ = sim_env.step(u)
        total_reward += reward
        if done:
            break
    return -total_reward  # negative because we will minimize

def run_mpc(env, horizon=10):
    """
    Run MPC on the environment using finite-horizon optimization.
    Returns state trajectory and applied actions.
    """
    state = env.reset()
    done = False

    state_hist = [state.copy()]
    action_hist = []

    step_count = 0

    while not done:
        step_count += 1
        # Initial guess: constant infusion
        u_guess = 0.5 * np.ones(horizon)

        # Optimize over the horizon
        res = minimize(simulate_horizon,
                       u_guess,
                       args=(env, state),
                       bounds=[(0, 1)] * horizon,
                       method='L-BFGS-B',
                       options={'maxiter': 50})

        u_opt = res.x
        action = float(u_opt[0])  # apply only the first action

        # Step environment
        next_state, reward, done, info = env.step(action)

        # Store
        state_hist.append(next_state.copy())
        action_hist.append(action)

        state = next_state

        # Print progress
        print(f"Step {step_count:3d} | t={info['t']:.2f} | applied u={action:.3f} "
              f"| reward={reward:.3f} | done={done}")

    return np.array(state_hist), np.array(action_hist)


def plot_trajectory(state_hist, action_hist, dt=0.05):
    """
    Plot the trajectories: cancer, bone marrow, and control action.
    """
    t = np.arange(state_hist.shape[0]) * dt

    pc = state_hist[:, 0]
    qc = state_hist[:, 1]
    cc = state_hist[:, 2]

    pb = state_hist[:, 3]
    qb = state_hist[:, 4]
    cb = state_hist[:, 5]

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Cancer
    axs[0].plot(t, pc, label='P_cancer')
    axs[0].plot(t, qc, label='Q_cancer')
    axs[0].plot(t, cc, label='C_cancer')
    axs[0].set_ylabel("Cancer")
    axs[0].legend()
    axs[0].grid(True)

    # Bone marrow
    axs[1].plot(t, pb, label='P_bm')
    axs[1].plot(t, qb, label='Q_bm')
    axs[1].plot(t, cb, label='C_bm')
    axs[1].set_ylabel("Bone Marrow")
    axs[1].legend()
    axs[1].grid(True)

    # Control action
    t_action = t[:-1]  # action applied at start of step
    axs[2].step(t_action, action_hist, where='post')
    axs[2].set_ylabel("Infusion Rate (u)")
    axs[2].set_xlabel("Time [days]")
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    env = ChemotherapyEnv()
    horizon = 10  # MPC lookahead steps

    print("Starting MPC simulation...")
    state_hist, action_hist = run_mpc(env, horizon=horizon)
    print("Simulation complete. Plotting results...")
    plot_trajectory(state_hist, action_hist, dt=env.dt)
