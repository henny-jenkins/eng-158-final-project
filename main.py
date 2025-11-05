from dynamics import dynamics_pd, dynamics_pd_pk
from dataclasses import dataclass, field
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp, OdeSolution   # for the dynamics simulation
from scipy.interpolate import interp1d
import numpy as np

def calc_rho(t_half_hrs: float):
    """
    this method calculates the drug clearance rate rho [days^-1] from a drug's half life [hrs]
    """
    t_half_days = t_half_hrs / 24.0
    rho = np.log(2) / t_half_days
    return rho

@dataclass
class CellPopulationParams:
    """
    dataclass to represent the exhaustive parameter set for an arbitrary cell population in this dynamical model
    """
    gamma: float            # proliferative cells self-renew coeff.
    delta: float            # proliferative cell die off coeff.
    alpha: float            # proliferative -> quiscent transition coeff.
    beta: float             # quiescent -> proliferative transition coeff.
    lambda_param: float     # quiescent cell die off coeff.
    rho_star_p: float       # for calculating initial conditions
    initial_conditions: np.ndarray = field(init=False)

    def __post_init__(self):
        """
        calculate population initial conditions [p0, q0, c0] based on rho_star_p, per Eastman et al. 2021
        """
        self.initial_conditions = np.array([self.rho_star_p, 1 - self.rho_star_p, 0])

@dataclass
class SimParams:
    """
    dataclass to represent all other simulation params
    """
    s: float                # dosing strength parameter
    rho: float | None       # drug clearance rate

@dataclass
class AnalysisPackage:
    """
    dataclass to bundle both copies of the dynamics simulation with drug schedule
    """
    bone_marrow_soln: OdeSolution   # simulation from solve_ivp
    cancer_soln: OdeSolution        # simulation from solve_ivp
    drug_schedule: np.ndarray       # time-varying control signal


def main():
    # instantiate parameters for each cell population from Eastman et al. 2021
    bone_marrow_params = CellPopulationParams(
            gamma=1.470,
            delta=0.000,
            alpha=5.643,
            beta=0.480,
            lambda_param=0.164,
            rho_star_p=0.103
            )
    breast_cancer_params = CellPopulationParams(
            gamma=0.500,
            delta=0.477,
            alpha=0.218,
            beta=0.050,
            lambda_param=0.000,
            rho_star_p=0.200
            )
    ovarian_cancer_params = CellPopulationParams(
            gamma=0.6685,
            delta=0.4597,
            alpha=0.2225,
            beta=0.0500,
            lambda_param=0.0000,
            rho_star_p=0.3600
            )

    # define simulation configuration
    SIMULATION_TIME = 21    # [days]
    SIM_STEPS = 1000
    t_span = (0, 21)
    t_eval = np.linspace(*t_span, SIM_STEPS)
    u_input = 0.5 * np.sin(t_eval) + 0.5   # arbitrary control signal for now — would obtain from optimizer for MPC or policy for RL
    u_input_unforced = 0 * t_eval
    u = interp1d(t_eval, u_input, fill_value="extrapolate")    # pass control signal as anonymous func
    u_unforced = interp1d(t_eval, u_input_unforced, fill_value="extrapolate")

    # select chemotherapeutic agent (somewhat arbitrary)
    T_HALF_HRS_PACLITAXEL = 10           # [hrs]
    rho_paclitaxel = calc_rho(T_HALF_HRS_PACLITAXEL)    # [days^-1]
    sim_params = SimParams(s=1, rho=rho_paclitaxel)

    # simulate the PD system under arbitrary control signal — 2 copies of 2D dynamical system (one for healthy population, one for cancerous population)
    sol_bm = solve_ivp(dynamics_pd,
                       t_span,
                       bone_marrow_params.initial_conditions[0:-1],
                       t_eval=t_eval,
                       args = (u, bone_marrow_params, sim_params))
    sol_breast_cancer = solve_ivp(dynamics_pd,
                       t_span,
                       breast_cancer_params.initial_conditions[0:-1],
                       t_eval=t_eval,
                       args = (u, breast_cancer_params, sim_params))
    breast_cancer_simulation_results = AnalysisPackage(
            bone_marrow_soln=sol_bm,
            cancer_soln=sol_breast_cancer,
            drug_schedule=u_input
            )

    # simulate PD unforced
    sol_bm_unforced = solve_ivp(dynamics_pd,
                       t_span,
                       bone_marrow_params.initial_conditions[0:-1],
                       t_eval=t_eval,
                       args = (u_unforced, bone_marrow_params, sim_params))
    sol_breast_cancer_unforced = solve_ivp(dynamics_pd,
                       t_span,
                       breast_cancer_params.initial_conditions[0:-1],
                       t_eval=t_eval,
                       args = (u_unforced, breast_cancer_params, sim_params))
    breast_cancer_simulation_results_unforced = AnalysisPackage(
            bone_marrow_soln=sol_bm_unforced,
            cancer_soln=sol_breast_cancer_unforced,
            drug_schedule=u_input_unforced
            )

    # simulate PK under arbitrary control signal
    sol_bm_pk = solve_ivp(dynamics_pd_pk,
                       t_span,
                       bone_marrow_params.initial_conditions,
                       t_eval=t_eval,
                       args = (u, bone_marrow_params, sim_params))
    sol_breast_cancer_pk = solve_ivp(dynamics_pd_pk,
                       t_span,
                       breast_cancer_params.initial_conditions,
                       t_eval=t_eval,
                       args = (u, breast_cancer_params, sim_params))
    breast_cancer_simulation_results_pk = AnalysisPackage(
            bone_marrow_soln=sol_bm_pk,
            cancer_soln=sol_breast_cancer_pk,
            drug_schedule=u_input
            )

    # plot all simulations
    plot_sim(breast_cancer_simulation_results_unforced)
    plot_sim(breast_cancer_simulation_results)
    plot_sim(breast_cancer_simulation_results_pk)

def plot_sim(sim_results: AnalysisPackage):
    """
    function to handle plotting of results from a single simulation
    """

    # pull out simulation data as local variables
    time = sim_results.cancer_soln.t
    cancer_p = sim_results.cancer_soln.y[0, :]
    cancer_q = sim_results.cancer_soln.y[1, :]
    bonemarrow_p = sim_results.bone_marrow_soln.y[0, :]
    bonemarrow_q = sim_results.bone_marrow_soln.y[1, :]
    u = sim_results.drug_schedule
    
    try:
        is_pk = True
        cancer_c = sim_results.cancer_soln.y[2, :]
        bonemarrow_c = sim_results.bone_marrow_soln.y[2, :]
    except:
        print("no concentration signal detected")
        is_pk = False

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # cancer plot
    axes[0].plot(time, cancer_p, label="P_cancer")
    axes[0].plot(time, cancer_q, label="Q_cancer")
    axes[0].plot(time, u, label="Drug Schedule")
    axes[0].set_title("Cancer Cells Under Drug Schedule")
    axes[0].set_xlabel("Time [days]")
    axes[0].set_ylabel("Cell Density / Infusion Fraction")

    # healthy cell plot
    axes[1].plot(time, bonemarrow_p, label="P_bm")
    axes[1].plot(time, bonemarrow_q, label="Q_bm")
    axes[1].plot(time, u, label="Drug Schedule")
    axes[1].set_title("Bone Marrow Cells Under Drug Schedule")
    axes[1].set_xlabel("Time [days]")
    axes[1].set_ylabel("Cell Density / Infusion Fraction")

    if is_pk:
        # plot the concentration state if it's available
        axes2 = axes[0].twinx()
        axes2.plot(time, cancer_c, color="k", label="C_cancer")
        axes2.set_ylabel("Relative Drug Concentration")
        axes2.legend(loc="upper right")
        axes3 = axes[1].twinx()
        axes3.plot(time, bonemarrow_c, color="k", label="C_bm")
        axes3.set_ylabel("Relative Drug Concentration")
        axes3.legend(loc="upper right")

    axes[0].legend(loc='upper left')
    axes[1].legend(loc='upper left')
    plt.show()


if __name__ == "__main__":
    main()
