# from dynamics import dynamics_pd, dynamics_pd_pk
from dataclasses import dataclass
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp   # for the dynamics simulation
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
    pass

if __name__ == "__main__":
    main()
