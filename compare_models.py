import numpy as np
import matplotlib.pyplot as plt
from chemo_env import ChemotherapyEnv
from evaluate_policy import evaluate
from mpc import run_mpc


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
