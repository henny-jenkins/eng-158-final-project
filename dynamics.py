import numpy as np

def dynamics_pd(t, x_vec, u, cell_params, sim_params):
    """
    dynamics model with the original pharmacodynamic models from Eastman et al. 2021
    """

    # specify state variables
    p = x_vec[0]
    q = x_vec[1]

    # specify model parameters
    gamma = cell_params.gamma
    delta = cell_params.delta
    alpha = cell_params.alpha
    beta = cell_params.beta
    lambda_param = cell_params.lambda_param
    s = sim_params.s
    u_val = u(t)    # u is an anonymous func, so need to query @ time=t to get float

    # return state derivatives
    p_dot = ((gamma - delta - alpha - (s * u_val)) * p) + (beta * q)
    q_dot = (alpha * p) - ((beta + lambda_param) * q)
    x_dot = np.array([p_dot, q_dot])
    return x_dot

def dynamics_pd_pk(t, x_vec, u, cell_params, sim_params):
    """
    extended PD model to include PK. In this model, u represents a fraction of a max infusion rate and c represents a relative concentration of a chemotherapeutic agent that clears from the body on the timescale of realistically used chemotherapeutic drugs
    """

    # specify state variables
    p = x_vec[0]
    q = x_vec[1]
    c = x_vec[2]

    # specify model parameters
    gamma = cell_params.gamma
    delta = cell_params.delta
    alpha = cell_params.alpha
    beta = cell_params.beta
    lambda_param = cell_params.lambda_param
    s = sim_params.s
    rho = sim_params.rho
    u_val = u(t)    # u is an anonymous func, so need to query @ time=t to get float

    # return state derivatives
    p_dot = ((gamma - delta - alpha - (s * c)) * p) + (beta * q)
    q_dot = (alpha * p) - ((beta + lambda_param) * q)
    c_dot = (- rho * c) + u_val
    x_dot = np.array([p_dot, q_dot, c_dot])
    return x_dot
