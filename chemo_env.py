# chemo_env.py

import numpy as np
from dataclasses import dataclass
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from dynamics import dynamics_pd_pk  # uses your existing implementation


def calc_rho(t_half_hrs: float) -> float:
    """
    Calculate drug clearance rate rho [days^-1] from drug half-life [hrs].
    """
    t_half_days = t_half_hrs / 24.0
    rho = np.log(2.0) / t_half_days
    return float(rho)


@dataclass
class CellPopulationParams:
    """
    Parameters for a single cell population (cancer or bone marrow).
    """
    gamma: float
    delta: float
    alpha: float
    beta: float
    lambda_param: float
    rho_star_p: float  # for initial conditions

    @property
    def initial_conditions_pdpk(self) -> np.ndarray:
        """
        Return [p0, q0, c0] initial conditions.
        """
        return np.array([self.rho_star_p, 1.0 - self.rho_star_p, 0.0], dtype=float)


@dataclass
class SimParams:
    """
    Simulation parameters shared across populations.
    """
    s: float                 # dosing strength parameter
    rho: float               # drug clearance rate


class ChemotherapyEnv:
    """
    Continuous-time chemotherapeutic dosing environment using PD+PK model.

    State: [P_cancer, Q_cancer, C_cancer, P_bm, Q_bm, C_bm]
    Action: scalar u in [0, 1] (infusion rate)
    """

    def __init__(
        self,
        dt: float = 0.05,          # time step [days]
        t_max: float = 21.0,       # horizon [days]
        t_half_hrs: float = 10.0,  # Paclitaxel half-life
        reward_lambda_toxic: float = 10.0,
        reward_lambda_u: float = 0.3,
        seed: int | None = None,
    ):
        self.rng = np.random.default_rng(seed)

        # PD parameters (from Eastman-style setup / your main.py)
        self.bone_marrow_params = CellPopulationParams(
            gamma=1.470,
            delta=0.000,
            alpha=5.643,
            beta=0.480,
            lambda_param=0.164,
            rho_star_p=0.103,
        )
        self.breast_cancer_params = CellPopulationParams(
            gamma=0.500,
            delta=0.477,
            alpha=0.218,
            beta=0.050,
            lambda_param=0.000,
            rho_star_p=0.200,
        )

        # Shared sim params (PK)
        rho_paclitaxel = calc_rho(t_half_hrs)
        self.sim_params = SimParams(s=1.0, rho=rho_paclitaxel)

        # Time config
        self.dt = float(dt)
        self.t_max = float(t_max)

        # Reward weights
        self.lambda_toxic = reward_lambda_toxic
        self.lambda_u = reward_lambda_u

        # Termination thresholds
        self.min_bm_fraction = 0.2    # if bone marrow P+Q < this, episode ends (toxicity)
        self.max_cancer_fraction = 2.0  # if cancer P+Q > this, treat as failure

        # Internal state
        self.t = 0.0
        self.state = None

    def reset(self) -> np.ndarray:
        """
        Reset episode to t=0 with initial populations.

        Returns:
            A normalized observation vector of shape (6,).
        """
        self.t = 0.0

        cancer_init = self.breast_cancer_params.initial_conditions_pdpk.copy()
        bm_init = self.bone_marrow_params.initial_conditions_pdpk.copy()

        # Optional small randomization:
        cancer_init[:2] *= self.rng.normal(loc=1.0, scale=0.02, size=2)
        bm_init[:2] *= self.rng.normal(loc=1.0, scale=0.02, size=2)

        # self.state is a flat numpy array of shape (6,) representing the full state
        self.state = np.concatenate([cancer_init, bm_init]).astype(float)

        return self._normalize_state(self.state)

    def step(self, action: float):
        """
        Take one step given a scalar action in [0, 1].

        Returns:
            observation (np.ndarray): The full, normalized state vector of shape (6,).
            reward (float): The reward for the current step.
            done (bool): Whether the episode has terminated.
            info (dict): A dictionary with auxiliary information.
        """
        # Clip action to [0, 1]
        u_scalar = float(np.clip(action, 0.0, 1.0))

        # Make action a constant over [t, t+dt] for both populations
        t_span = (self.t, self.t + self.dt)
        t_eval = [self.t + self.dt]

        u = interp1d([self.t, self.t + self.dt], [u_scalar, u_scalar], fill_value="extrapolate")

        # Split current state
        cancer_state = self.state[0:3]
        bm_state = self.state[3:6]

        # Integrate dynamics for cancer
        cancer_sol = solve_ivp(
            dynamics_pd_pk,
            t_span,
            cancer_state,
            t_eval=t_eval,
            args=(u, self.breast_cancer_params, self.sim_params),
            rtol=1e-6,
            atol=1e-8,
        )
        # Integrate dynamics for bone marrow
        bm_sol = solve_ivp(
            dynamics_pd_pk,
            t_span,
            bm_state,
            t_eval=t_eval,
            args=(u, self.bone_marrow_params, self.sim_params),
            rtol=1e-6,
            atol=1e-8,
        )

        # TODO: fix the state definition
        new_cancer_state = cancer_sol.y[:, -1]
        new_bm_state = bm_sol.y[:, -1]
        new_state = np.concatenate([new_cancer_state, new_bm_state])

        # Clamp to non-negative to avoid tiny numerical negatives
        new_state = np.maximum(new_state, 0.0)

        self.state = new_state
        self.t += self.dt

        reward = self._compute_reward(self.state, u_scalar)
        done, info = self._check_done(self.state)

        return self._normalize_state(self.state), float(reward), bool(done), info

    def _compute_reward(self, state: np.ndarray, u_scalar: float) -> float:
        """
        Reward encourages tumor suppression, bone marrow preservation,
        and penalizes drug usage.
        """
        pc, qc, cc, pb, qb, cb = state

        cancer_load = pc + qc
        bm_load = pb + qb

        # Main components
        tumor_penalty = cancer_load  # we want this small
        bm_reward = bm_load          # we want this large

        drug_penalty = u_scalar ** 2

        reward = -tumor_penalty + bm_reward - self.lambda_u * drug_penalty

        return float(reward)

    def _check_done(self, state: np.ndarray):
        """
        Check termination conditions.
        """
        pc, qc, cc, pb, qb, cb = state
        cancer_load = pc + qc
        bm_load = pb + qb

        done = False
        info = {}

        if self.t >= self.t_max:
            done = True
            info["terminal_reason"] = "time_limit"

        if bm_load < self.min_bm_fraction:
            done = True
            info["terminal_reason"] = "severe_toxicity"

        if cancer_load > self.max_cancer_fraction:
            done = True
            info["terminal_reason"] = "uncontrolled_tumor"

        return done, info

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """
        Basic normalization: assume P,Q in [0, 1], C in [0, ~1/rho].
        """
        pc, qc, cc, pb, qb, cb = state

        max_pq = 1
        max_c = 1.0 / self.sim_params.rho  # rough steady-state level

        norm = np.array([
            pc / max_pq,
            qc / max_pq,
            cc / max_c,
            pb / max_pq,
            qb / max_pq,
            cb / max_c,
        ], dtype=float)

        return norm
