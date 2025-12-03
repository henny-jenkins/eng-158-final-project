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


def plot_multiple_b(state_histories, action_histories, b_values, dt=0.05):
    """
    Plot multiple MPC runs with different reward_b values in columns.
    
    Parameters
    ----------
    state_histories : list of np.ndarray
        Each element is (T,6) state history for one b value.
    action_histories : list of np.ndarray
        Each element is (T-1,) applied actions.
    b_values : list of float
        The reward_b values used for each run.
    dt : float
        Time step size.
    """
    n_cols = len(b_values)
    n_rows = 3  # cancer, bone marrow, control

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 8), sharex=True)
    
    for col, (state_hist, action_hist, b) in enumerate(zip(state_histories, action_histories, b_values)):
        t = np.arange(state_hist.shape[0]) * dt

        pc, qc, cc = state_hist[:,0], state_hist[:,1], state_hist[:,2]
        pb, qb, cb = state_hist[:,3], state_hist[:,4], state_hist[:,5]

        # Cancer
        axs[0, col].plot(t, pc, label='P_cancer')
        axs[0, col].plot(t, qc, label='Q_cancer')
        axs[0, col].plot(t, cc, label='C_cancer')
        axs[0, col].set_ylabel("Cancer" if col == 0 else "")
        axs[0, col].legend()
        axs[0, col].grid(True)

        # Bone marrow
        axs[1, col].plot(t, pb, label='P_bm')
        axs[1, col].plot(t, qb, label='Q_bm')
        axs[1, col].plot(t, cb, label='C_bm')
        axs[1, col].set_ylabel("Bone Marrow" if col == 0 else "")
        axs[1, col].legend()
        axs[1, col].grid(True)

        # Control action
        t_action = t[:-1]
        axs[2, col].step(t_action, action_hist, where='post')
        axs[2, col].set_ylabel("Infusion Rate (u)" if col == 0 else "")
        axs[2, col].set_xlabel("Time [days]")
        axs[2, col].grid(True)

        # Add column title
        axs[0, col].set_title(f"reward_b = {b}")

    plt.tight_layout()
    plt.savefig("mpc.png")
    plt.show()



if __name__ == "__main__":
    b_values = [0.02, 0.05, 0.25]
    state_histories = []
    action_histories = []

    for b in b_values:
        env = ChemotherapyEnv(reward_b=b)
        state_hist, action_hist = run_mpc(env, horizon=10)
        state_histories.append(state_hist)
        action_histories.append(action_hist)

    plot_multiple_b(state_histories, action_histories, b_values)
