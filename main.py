from dynamics import dynamics_pd, dynamics_pd_pk
from dataclasses import dataclass
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp   # for the dynamics simulation
from scipy.interpolate import interp1d
import numpy as np

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

@dataclass
class SimParams:
    """
    dataclass to represent all other simulation params
    """
    s: float                # dosing strength parameter
    rho: float | None       # drug clearance rate

def main():
    # instantiate parameters for each cell population from Eastman et al. 2021
    bone_marrow_params = CellPopulationParams(
            gamma=1.470,
            delta=0.000,
            alpha=5.643,
            beta=0.480,
            lambda_param=0.164
            )
    breast_cancer_params = CellPopulationParams(
            gamma=0.500,
            delta=0.477,
            alpha=0.218,
            beta=0.050,
            lambda_param=0.000
            )
    ovarian_cancer_params = CellPopulationParams(
            gamma=0.6685,
            delta=0.4597,
            alpha=0.2225,
            beta=0.0500,
            lambda_param=0.0000
            )

    # define simulation configuration
    SIMULATION_TIME = 21    # [days?]
    SIM_STEPS = 1000
    t_input = np.linspace(0, SIMULATION_TIME, SIM_STEPS)
    u_input = np.sin(t_input)   # arbitrary control signal for now — would obtain from optimizer for MPC or policy for RL
    u = interp1d(t_input, u_input, fill_value="extrapolate")
    print("everything working so far")

if __name__ == "__main__":
    main()
