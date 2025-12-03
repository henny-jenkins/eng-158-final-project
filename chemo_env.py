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
        reward_b: float = 0.05,        # bone marrow vs cancer balance factor (must be non-negative)
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
        self.reward_b = reward_b

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
        t_eval = [self.t, self.t + self.dt] # eval at both endpoints for integral

        u = interp1d([self.t, self.t + self.dt], [u_scalar, u_scalar], fill_value="extrapolate")

        # Split current state
        cancer_state = self.state[0:3].copy()
        bm_state = self.state[3:6].copy()

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

        # extract start / end values for each compartment
        # cancer_sol.y and bm_sol.y shapes: (3, 2)
        cancer_start = cancer_sol.y[:, 0]
        cancer_end = cancer_sol.y[:, -1]
        bm_start = bm_sol.y[:, 0]
        bm_end = bm_sol.y[:, -1]

        # set new (flat) state to the end-of-step values
        self.state = np.concatenate([cancer_end, bm_end]).astype(float)

        # advance time
        self.t += self.dt

        # compute reward using trapezoidal approximation over the step:
        # integral_{t}^{t+dt} [P_bm(s)+Q_bm(s)] ds  ≈ dt * 0.5*( (P_bm+Q_bm)_start + (P_bm+Q_bm)_end )
        reward = self._compute_reward(bm_start, bm_end, u_scalar, self.dt)

        done, info = self._check_done(self.state)

        # helpful debugging info
        info.update({
            "t": self.t,
            "u": u_scalar,
            "bm_start": bm_start.copy(),
            "bm_end": bm_end.copy(),
            "cancer_end": cancer_end.copy(),
        })

        return self._normalize_state(self.state), float(reward), bool(done), info

    def _compute_reward(self, bm_start: np.ndarray, bm_end: np.ndarray, u_scalar: float, dt: float) -> float:
        """
        Implements the immediate reward from the paper (Eq. (4)):
        R(s_t,a_t) = ∫_{t}^{t+dt} [ P_bm(s) + Q_bm(s) - (b/2)*(1-a)^2 ] ds

        Here we approximate the integral for the state-dependent part with the trapezoid rule,
        and multiply the constant action penalty by dt.

        Parameters
        ----------
        bm_start : array-like, shape (3,)
            bone marrow [P, Q, C] at the start of the step
        bm_end : array-like, shape (3,)
            bone marrow [P, Q, C] at the end of the step
        u_scalar : float
            scalar action in [0,1]
        dt : float
            step duration in days

        Returns
        -------
        reward : float
        """

        # extract proliferative + quiescent fractions at start and end
        pb0, qb0, _cb0 = bm_start
        pb1, qb1, _cb1 = bm_end

        bm_val_start = float(pb0 + qb0)
        bm_val_end = float(pb1 + qb1)

        # trapezoid approx of integral of bone-marrow preservation term
        bm_integral = dt * 0.5 * (bm_val_start + bm_val_end)

        # action penalty from paper: (b/2) * (1 - a)^2 integrated over dt
        b = float(self.reward_b)  # paper's "b" weight
        action_penalty_integral = dt * (0.5 * b * (1.0 - u_scalar) ** 2)

        reward = bm_integral - action_penalty_integral

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
