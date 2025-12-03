import numpy as np
import matplotlib.pyplot as plt
from chemo_env import ChemotherapyEnv
from evaluate_policy import evaluate
from mpc import run_mpc

def compute_objective(env, states, actions, return_per_step=False):
    """
    Compute Eastman-style objective:
        J = sum_t [ P_bm + Q_bm - (b/2)(1-u)^2 ] * dt
    Robust to off-by-one: expects len(states) >= len(actions) and uses actions[i]
    for the transition states[i] -> states[i+1].

    If len(actions) == len(states), this will drop the last action (with a warning).
    """
    states = np.asarray(states)
    actions = np.asarray(actions)

    n_states = states.shape[0]
    n_actions = actions.shape[0]

    if n_states < 2:
        raise ValueError("states must contain at least two timesteps to form one transition")

    # Expected: n_states == n_actions + 1
    if n_actions == n_states:
        # Common mismatch: one action per state. We'll drop the final action so actions align with transitions.
        print("Warning: len(actions) == len(states). Dropping last action to align transitions.")
        actions = actions[:-1]
        n_actions -= 1
    elif n_actions > n_states - 1:
        # More actions than transitions -> truncate actions
        print("Warning: more actions than transitions; truncating actions to match transitions.")
        actions = actions[: n_states - 1]
        n_actions = actions.shape[0]
    elif n_actions < n_states - 1:
        # Fewer actions than transitions -> truncate states to match
        print("Warning: fewer actions than transitions; truncating states to match actions.")
        states = states[: n_actions + 1]
        n_states = states.shape[0]

    # Now n_states == n_actions + 1
    dt = float(env.dt)
    b = float(env.reward_b)

    per_step_rewards = np.empty(n_actions, dtype=float)

    for i in range(n_actions):
        bm_start = states[i, 3:5]   # Pb, Qb at start
        bm_end = states[i+1, 3:5]   # Pb, Qb at end
        u = float(actions[i])

        bm_val_start = float(bm_start[0] + bm_start[1])
        bm_val_end = float(bm_end[0] + bm_end[1])

        bm_integral = dt * 0.5 * (bm_val_start + bm_val_end)
        action_penalty_integral = dt * (0.5 * b * (1.0 - u) ** 2)

        per_step_rewards[i] = bm_integral - action_penalty_integral

    J = float(np.sum(per_step_rewards))
    if return_per_step:
        return J, per_step_rewards
    return J


if __name__ == "__main__":
    env = ChemotherapyEnv()     # define env

    # run ppo
    states_ppo, actions_ppo = evaluate(env)
    t_ppo = np.arange(len(states_ppo)) * env.dt

    # run mpc with same env
    states_mpc, actions_mpc = run_mpc(env)
    t_mpc = np.arange(len(states_mpc)) * env.dt

    # Unpack PPO states
    Pc_ppo, Qc_ppo, Cc_ppo = states_ppo[:,0], states_ppo[:,1], states_ppo[:,2]
    Pb_ppo, Qb_ppo, Cb_ppo = states_ppo[:,3], states_ppo[:,4], states_ppo[:,5]

    Pc_mpc, Qc_mpc, Cc_mpc = states_mpc[:,0], states_mpc[:,1], states_mpc[:,2]
    Pb_mpc, Qb_mpc, Cb_mpc = states_mpc[:,3], states_mpc[:,4], states_mpc[:,5]


    # --- compute objective functionals ---
    J_ppo = compute_objective(env, states_ppo, actions_ppo)
    J_mpc = compute_objective(env, states_mpc, actions_mpc)

    print("\n===== Objective Functional Scores =====")
    print(f"PPO objective J = {J_ppo:.4f}")
    print(f"MPC objective J = {J_mpc:.4f}")
    print("=======================================\n")

    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # plot ppo
    axes[0].plot(t_ppo, Pc_ppo, 'b',  label="PPO P_c")
    axes[0].plot(t_ppo, Qc_ppo, 'r', label="PPO Q_c")
    axes[0].plot(t_ppo, Cc_ppo, 'g', label="PPO C_c")

    # plot mpc
    axes[0].plot(t_mpc, Pc_mpc, 'b--', label="MPC P_c")
    axes[0].plot(t_mpc, Qc_mpc, 'r--', label="MPC Q_c")
    axes[0].plot(t_mpc, Cc_mpc, 'g--', label="MPC C_c")

    # configure
    axes[0].set_title(f"Cancer Cell Dynamics, b={env.reward_b}")
    axes[0].set_xlabel("Time (days)")
    axes[0].legend()

    # plot ppo
    axes[1].plot(t_ppo, Pb_ppo, 'b', label="PPO P_bm")
    axes[1].plot(t_ppo, Qb_ppo, 'r', label="PPO Q_bm")
    axes[1].plot(t_ppo, Cb_ppo, 'g', label="PPO C_bm")

    # plot mpc
    axes[1].plot(t_mpc, Pb_mpc, 'b--', label="MPC P_bm")
    axes[1].plot(t_mpc, Qb_mpc, 'r--', label="MPC Q_bm")
    axes[1].plot(t_mpc, Cb_mpc, 'g--', label="MPC C_bm")

    # configure
    axes[1].set_title("Bone Marrow Dynamics")
    axes[1].set_xlabel("Time (days)")
    axes[1].legend()

    axes[2].plot(t_ppo, actions_ppo, label="PPO Dosing u(t)")
    axes[2].plot(t_ppo, actions_mpc, 'r', label="MPC Dosing u(t)")
    axes[2].set_title("Optimal Drug Schedule")
    axes[2].set_xlabel("Time (days)")
    axes[2].set_ylabel("Infusion Rate")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("ppo_mpc_comp.png")
    plt.show()
