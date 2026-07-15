"""
calibration.py
...
"""

import numpy as np
from scipy.integrate import solve_ivp


def loglike(params, observed_days, observed_volumes, meas_sigma=5.0):
    """
    Log-likelihood of the observed (noisy) volumes given NSM parameters.
    Since this simplified version treats the model as deterministic
    (no process noise), we solve the underlying ODE directly with
    scipy's solve_ivp instead of a manual Euler-Maruyama loop -- much
    faster and more accurate for repeated MCMC evaluations.
    """
    a, b, alpha, V0 = params
    duration = max(observed_days)

    def ode_rhs(t, V):
        return a * V**alpha - b * V

    sol = solve_ivp(ode_rhs, [0, duration], [V0],
                     t_eval=observed_days, method="RK45")

    if not sol.success:
        return -np.inf

    predicted = sol.y[0]

    residuals = observed_volumes - predicted
    return -0.5 * np.sum((residuals / meas_sigma) ** 2)


def log_prior(params):
    a, b, alpha, V0 = params
    if not (0.1 < a < 5.0):
        return -np.inf
    if not (0.01 < b < 1.0):
        return -np.inf
    if not (0.3 < alpha < 0.99):
        return -np.inf
    if not (5.0 < V0 < 200.0):
        return -np.inf
    return 0.0


def log_posterior(params, observed_days, observed_volumes):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + loglike(params, observed_days, observed_volumes)



################################################
## POPULATION-LEVEL CALIBRATION (Milestone 2)
################################################
# Joint calibration on all 8 mice: a, b, alpha are shared population
# parameters (fixed effects, Eq. 20), while each mouse keeps its own
# V0 (random effect, Eq. 21). This pools information across mice,
# which the paper suggests can help resolve identifiability issues
# that appear when calibrating on a single mouse (Section III-A).

def loglike_population(params, mice_days, mice_volumes, meas_sigma=5.0):
    """
    params = [a, b, alpha, V0_1, V0_2, ..., V0_8]
    mice_days, mice_volumes: lists of arrays, one per mouse
    """
    a, b, alpha = params[0], params[1], params[2]
    V0_list = params[3:]

    total_ll = 0.0
    for V0, days, vols in zip(V0_list, mice_days, mice_volumes):
        duration = max(days)

        def ode_rhs(t, V):
            return a * V**alpha - b * V

        sol = solve_ivp(ode_rhs, [0, duration], [V0], t_eval=days, method="RK45")
        if not sol.success:
            return -np.inf

        predicted = sol.y[0]
        residuals = vols - predicted
        total_ll += -0.5 * np.sum((residuals / meas_sigma) ** 2)

    return total_ll


def log_prior_population(params):
    a, b, alpha = params[0], params[1], params[2]
    V0_list = params[3:]

    if not (0.1 < a < 5.0):
        return -np.inf
    if not (0.01 < b < 1.0):
        return -np.inf
    if not (0.3 < alpha < 0.99):
        return -np.inf
    for V0 in V0_list:
        if not (5.0 < V0 < 200.0):
            return -np.inf
    return 0.0


def log_posterior_population(params, mice_days, mice_volumes):
    lp = log_prior_population(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + loglike_population(params, mice_days, mice_volumes)
