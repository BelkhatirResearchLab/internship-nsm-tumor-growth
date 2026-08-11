"""
Extended Kalman Filter likelihood for the NSM model, using a Lamperti
transform (Z = ln V) so the noise term becomes state-independent and
a standard EKF predict/update cycle can be applied. Faster than the
Monte Carlo likelihood since it doesn't need repeated simulation, but
not yet validated against it.
"""

import numpy as np


def ekf_loglikelihood(params, observed_days, observed_volumes,
                       meas_sigma=5.0, dt=0.01):
    """
    params = (a, b, alpha, sigma, V0)

    State Z = ln(V), drift f(Z) = a*exp((alpha-1)*Z) - b - sigma^2/2,
    diffusion constant = sigma. Measurement function h(Z) = exp(Z),
    nonlinear, so it gets linearized at each update step.
    """
    a, b, alpha, sigma, V0 = params

    if V0 <= 0:
        return -np.inf

    m = np.log(V0)   # mean of Z
    P = 0.0           # variance of Z, V0 treated as known here

    t_current = 0.0
    total_loglik = 0.0

    for day, y_obs in zip(observed_days, observed_volumes):

        # predict: propagate mean/variance forward to the next
        # measurement day with a fine Euler step
        n_steps = max(int((day - t_current) / dt), 1)
        step = (day - t_current) / n_steps if n_steps > 0 else 0.0

        for _ in range(n_steps):
            f = a * np.exp((alpha - 1) * m) - b - 0.5 * sigma**2
            f_prime = a * (alpha - 1) * np.exp((alpha - 1) * m)
            m = m + f * step
            P = P + (2 * f_prime * P + sigma**2) * step
            P = max(P, 1e-10)

        t_current = day

        # update: correct using the new measurement
        h = np.exp(m)
        h_prime = np.exp(m)

        innovation = y_obs - h
        innovation_var = (h_prime**2) * P + meas_sigma**2

        if innovation_var <= 0:
            return -np.inf

        kalman_gain = P * h_prime / innovation_var

        m = m + kalman_gain * innovation
        P = (1 - kalman_gain * h_prime) * P
        P = max(P, 1e-10)

        total_loglik += -0.5 * (
            innovation**2 / innovation_var + np.log(2 * np.pi * innovation_var)
        )

    return total_loglik


def log_prior_ekf(params):
    a, b, alpha, sigma, V0 = params
    if not (0.1 < a < 5.0):
        return -np.inf
    if not (0.01 < b < 1.0):
        return -np.inf
    if not (0.3 < alpha < 0.99):
        return -np.inf
    if not (0.001 < sigma < 0.2):
        return -np.inf
    if not (5.0 < V0 < 200.0):
        return -np.inf
    return 0.0


def log_posterior_ekf(params, observed_days, observed_volumes):
    lp = log_prior_ekf(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ekf_loglikelihood(params, observed_days, observed_volumes)