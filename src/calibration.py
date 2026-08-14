"""
Parameter calibration for the uncontrolled stochastic NSM tumor
growth model. Includes a deterministic baseline likelihood, a Monte
Carlo likelihood that accounts for process noise, a KDE-based way to
pool individual per-mouse posteriors into a population prior, and a
sequential/online update mechanism.
"""

import numpy as np
from scipy.integrate import solve_ivp
from nsm_model import simulate_one_mouse


################################################
# deterministic likelihood (sigma = 0) — baseline / diagnostic
################################################

def loglike(params, observed_days, observed_volumes, meas_sigma=5.0):
    a, b, alpha, V0 = params
    duration = max(observed_days)

    def ode_rhs(t, V):
        return a * V**alpha - b * V

    sol = solve_ivp(ode_rhs, [0, duration], [V0],
                     t_eval=observed_days, method="RK45",
                     rtol=1e-4, atol=1e-4)

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
# Monte Carlo likelihood, single mouse — accounts for process noise
################################################
# instead of one deterministic curve, simulate several noisy
# trajectories for the same parameters and compare the data to the
# resulting mean/spread at each measurement day

def loglike_montecarlo(params, observed_days, observed_volumes,
                        meas_sigma=5.0, n_simulations=20, dt=0.1):
    a, b, alpha, sigma, V0 = params
    duration = max(observed_days)

    all_curves = np.zeros((n_simulations, len(observed_days)))
    for s in range(n_simulations):
        t_grid, V = simulate_one_mouse(a, b, alpha, beta=1.0, sigma=sigma,
                                         V0=V0, duration=duration, dt=dt)
        idx = np.searchsorted(t_grid, observed_days)
        all_curves[s] = V[idx]

    sim_mean = all_curves.mean(axis=0)
    sim_std = all_curves.std(axis=0)

    # process noise spread + measurement noise, added in quadrature
    total_std = np.sqrt(sim_std**2 + meas_sigma**2)

    residuals = observed_volumes - sim_mean
    return -0.5 * np.sum((residuals / total_std) ** 2 + 2 * np.log(total_std))


def log_prior_montecarlo(params):
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


def log_posterior_montecarlo(params, observed_days, observed_volumes):
    lp = log_prior_montecarlo(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + loglike_montecarlo(params, observed_days, observed_volumes)


################################################
# pooling individual posteriors into a population prior (KDE)
################################################
# each mouse gets calibrated on its own with loglike_montecarlo above
# (see notebook for the loop); this takes the resulting samples and
# builds a population-level prior via a kernel density estimate,
# Silverman bandwidth, expanded by a factor beta so it isn't too tight

def build_population_prior_kde(list_of_posterior_samples, bounds=None, beta=2.0, n_output_samples=2000):
    """
    list_of_posterior_samples: one array per mouse, each of shape
    (n_samples_i, n_params) -- columns (a, b, alpha, sigma) or
    (a, b, alpha, sigma, V0).
    bounds: list of (low, high) tuples, one per parameter, matching
    the columns above. Samples outside these bounds are rejected and
    redrawn, following Browning et al.'s approach.
    """
    X = np.vstack(list_of_posterior_samples)
    n, d = X.shape

    if bounds is None:
        bounds = [(0.1, 5.0), (0.01, 1.0), (0.3, 0.99), (0.001, 0.2), (5.0, 200.0)][:d]

    std_per_dim = np.std(X, axis=0)
    bandwidth = (4 / (d + 2))**(1 / (d + 4)) * n**(-1 / (d + 4)) * std_per_dim

    output = np.zeros((n_output_samples, d))
    for i in range(n_output_samples):
        while True:
            idx = np.random.choice(n)
            candidate = X[idx] + beta * bandwidth * np.random.randn(d)
            if all(bounds[k][0] < candidate[k] < bounds[k][1] for k in range(d)):
                output[i] = candidate
                break

    return output


################################################
# sequential / online update
################################################
# for each sample of (a, b, alpha) from a population prior, plus a V0,
# compute the cumulative log-likelihood as measurements come in one at
# a time; samples that keep matching the data stay weighted higher.
# a weighted average of the simulated curves at any point in time
# gives an updated prediction using only what's been observed so far.
#
# still uses the deterministic forward model, and V0_samples needs to
# be supplied separately since the pooled prior above only covers
# (a, b, alpha, sigma). not tested yet on a mouse held out of the pool.

def get_weights_nsm(param_samples, V0_samples, observed_days, observed_volumes,
                     meas_sigma=5.0):
    n_samples = len(param_samples)
    n_times = len(observed_days)

    all_curves = np.full((n_samples, n_times), np.nan)
    ll_contributions = np.full((n_samples, n_times), -np.inf)

    for i, (a, b, alpha) in enumerate(param_samples):
        V0 = V0_samples[i]

        def ode_rhs(t, V):
            return a * V**alpha - b * V

        sol = solve_ivp(ode_rhs, [0, max(observed_days)], [V0],
                         t_eval=observed_days, method="RK45",
                         rtol=1e-4, atol=1e-4)

        if not sol.success:
            continue

        predicted = sol.y[0]
        all_curves[i] = predicted

        residuals = observed_volumes - predicted
        ll_contributions[i] = -0.5 * (residuals / meas_sigma) ** 2

    cum_ll = np.cumsum(ll_contributions, axis=1)

    weights = np.zeros_like(cum_ll)
    for t in range(n_times):
        ll_t = cum_ll[:, t]
        ll_t = ll_t - np.nanmax(ll_t)
        w = np.exp(ll_t)
        weights[:, t] = w / np.nansum(w)

    return weights, all_curves