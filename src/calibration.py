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
