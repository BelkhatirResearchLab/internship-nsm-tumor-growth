"""
calibration.py

Calibration / parameter identification for the uncontrolled stochastic
NSM model, adapting the pseudo-hierarchical Bayesian method of
Browning et al. (2024, "Predicting Radiotherapy Patient Outcomes with
Real-Time Clinical Data...") to the NSM tumor growth model
(Belkhatir et al., 2020).

STATUS (matches notebooks/calibration_pipeline.py numbering):
- Milestone 1 (loglike/log_prior/log_posterior): single-mouse
  calibration with a SIMPLIFIED likelihood (deterministic model,
  sigma=0 -- process/dynamical noise is IGNORED). Kept only as a
  diagnostic: it shows biased parameter estimates when the data
  actually contain process noise (see docs/adaptation_plan.md and
  results/milestone1_deterministic_bias.png). Not a final result.
- Milestone 2 (loglike_montecarlo/...): single-mouse calibration with
  a MONTE CARLO likelihood that properly accounts for process noise
  (sigma > 0, estimated jointly with a, b, alpha). WORKING. Shows
  sigma and V0 converge well; a, b, alpha show an identifiability
  issue on a single mouse (see notebook diagnostic).
- Milestone 3 (loglike_montecarlo_population/...): population joint
  calibration, Monte Carlo likelihood, a/b/alpha/sigma shared across
  mice (fixed effects, Eq. 20-21 of the NSM paper), individual V0 per
  mouse. WORKING, in progress -- testing whether pooling more mice
  improves the identifiability issue seen in Milestone 2.
- get_weights_nsm: online/sequential update mechanism, adapted from
  Browning et al.'s get_weights() (analysis/inference.jl). WORKING,
  first version, but still based on the deterministic-likelihood
  population posterior (old Milestone 2, now removed from the
  notebook) -- needs to be re-run using the Milestone 3 (Monte Carlo)
  population posterior instead. Also not yet tested on a held-out
  mouse (true out-of-sample prediction).

NOTE: this is a mixed-effect joint calibration approach (shared fixed
effects across all mice in one MCMC), which is the NSM paper's own
formulation -- NOT Browning et al.'s approach (which calibrates each
patient separately and pools posteriors afterwards via KDE). See
docs/adaptation_plan.md for the open question on which approach to
prioritize going forward.

NEXT STEP: implement a likelihood that properly accounts for
process/dynamical noise via an Extended Kalman Filter (Lamperti
transform, as in Belkhatir et al. Section III-B), which would replace
the Monte Carlo likelihood with a much faster (but more complex)
alternative.
"""

import numpy as np
from scipy.integrate import solve_ivp
from nsm_model import simulate_one_mouse


################################################
## MILESTONE 1 — deterministic likelihood (diagnostic only)
################################################
# Treats the NSM model as a deterministic ODE (process noise sigma=0)
# with additive, constant-variance measurement noise (Eq. 20 of the
# NSM paper). Kept to demonstrate the bias this simplification causes
# -- not used for any final result.

def loglike(params, observed_days, observed_volumes, meas_sigma=5.0):
    """
    Log-likelihood of the observed (noisy) volumes given NSM parameters
    (a, b, alpha, V0), for a single mouse. Forward model solved as a
    deterministic ODE (sigma=0) with scipy's solve_ivp.
    """
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
    """Uniform prior on (a, b, alpha, V0)."""
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
## MILESTONE 2 — Monte Carlo likelihood, single mouse
################################################
# Instead of comparing observed data to a SINGLE deterministic
# trajectory (sigma=0), simulate many noisy trajectories (with the
# true sigma) for the same parameters, and compare the observed data
# to the resulting distribution (mean + spread) at each measurement
# day. This properly accounts for process noise, unlike loglike().

def loglike_montecarlo(params, observed_days, observed_volumes,
                        meas_sigma=5.0, n_simulations=20, dt=0.1):
    """
    params = (a, b, alpha, sigma, V0). Unlike loglike(), sigma
    (process noise) is now part of the parameters to estimate.
    """
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

    # total uncertainty = process noise spread + measurement noise,
    # combined in quadrature (variances add)
    total_std = np.sqrt(sim_std**2 + meas_sigma**2)

    residuals = observed_volumes - sim_mean
    return -0.5 * np.sum((residuals / total_std) ** 2 + 2 * np.log(total_std))


def log_prior_montecarlo(params):
    """Uniform prior on (a, b, alpha, sigma, V0)."""
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
## MILESTONE 3 — Monte Carlo likelihood, population (shared parameters)
################################################
# Same idea as loglike_montecarlo, but now a, b, alpha, sigma are
# SHARED across all mice (fixed effects), while each mouse keeps its
# own V0 (random effect). params = [a, b, alpha, sigma, V0_1, ..., V0_n]
# where n is the number of mice (variable, not fixed to 8).

def loglike_montecarlo_population(params, mice_days, mice_volumes,
                                    meas_sigma=5.0, n_simulations=8, dt=0.3):
    """
    n_simulations/dt kept lower than in loglike_montecarlo (single
    mouse) since this function is called once per mouse per
    evaluation -- with n mice that's already n times more simulations
    per MCMC step than the single-mouse version.
    """
    a, b, alpha, sigma = params[0], params[1], params[2], params[3]
    V0_list = params[4:]

    total_ll = 0.0

    for mouse_idx, V0 in enumerate(V0_list):
        days = mice_days[mouse_idx]
        vols = mice_volumes[mouse_idx]
        duration = max(days)

        all_curves = np.zeros((n_simulations, len(days)))
        for s in range(n_simulations):
            t_grid, V = simulate_one_mouse(a, b, alpha, beta=1.0, sigma=sigma,
                                             V0=V0, duration=duration, dt=dt)
            idx = np.searchsorted(t_grid, days)
            all_curves[s] = V[idx]

        sim_mean = all_curves.mean(axis=0)
        sim_std = all_curves.std(axis=0)
        total_std = np.sqrt(sim_std**2 + meas_sigma**2)

        residuals = vols - sim_mean
        total_ll += -0.5 * np.sum((residuals / total_std) ** 2 + 2 * np.log(total_std))

    return total_ll


def log_prior_montecarlo_population(params):
    """Uniform prior on (a, b, alpha, sigma, V0_1, ..., V0_n)."""
    a, b, alpha, sigma = params[0], params[1], params[2], params[3]
    V0_list = params[4:]

    if not (0.1 < a < 5.0):
        return -np.inf
    if not (0.01 < b < 1.0):
        return -np.inf
    if not (0.3 < alpha < 0.99):
        return -np.inf
    if not (0.001 < sigma < 0.2):
        return -np.inf
    for V0 in V0_list:
        if not (5.0 < V0 < 200.0):
            return -np.inf
    return 0.0


def log_posterior_montecarlo_population(params, mice_days, mice_volumes):
    lp = log_prior_montecarlo_population(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + loglike_montecarlo_population(params, mice_days, mice_volumes)


################################################
## ONLINE / SEQUENTIAL UPDATE
################################################
# Adapted from Browning et al.'s get_weights() (analysis/inference.jl).
# For each sample of (a, b, alpha) from a population posterior, and an
# associated V0, compute the CUMULATIVE log-likelihood as measurements
# arrive one at a time. This gives a weight per sample at each time
# step -- samples that keep fitting the incoming data stay heavily
# weighted, others get down-weighted. A weighted average of the
# simulated curves (using these weights) at any point in time gives an
# updated prediction using only the data seen so far -- the
# "real-time" mechanism adapted from the radiotherapy paper.
#
# CURRENT LIMITATIONS:
# - Still uses the deterministic forward model (no process noise in
#   the simulated curves themselves) -- should be re-run using
#   parameter samples from the Milestone 3 (Monte Carlo) population
#   posterior for consistency with the corrected likelihood.
# - Not yet tested on a held-out mouse (true out-of-sample prediction).

def get_weights_nsm(param_samples, V0_samples, observed_days, observed_volumes,
                     meas_sigma=5.0):
    """
    param_samples : array of shape (n_samples, 3), columns = (a, b, alpha)
    V0_samples    : array of shape (n_samples,), initial condition per sample
    observed_days, observed_volumes : the new mouse's measurements, in
        chronological order (as they "arrive" over time)

    Returns:
        weights    : array (n_samples, n_timepoints), normalized weights
                      at each time step (columns sum to 1)
        all_curves : array (n_samples, n_timepoints), simulated volume
                      trajectory of each sample at the observed days
    """
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

    # cumulative log-likelihood over time = sequential/online update
    cum_ll = np.cumsum(ll_contributions, axis=1)

    # normalized weights at each time step (log-sum-exp for stability)
    weights = np.zeros_like(cum_ll)
    for t in range(n_times):
        ll_t = cum_ll[:, t]
        ll_t = ll_t - np.nanmax(ll_t)
        w = np.exp(ll_t)
        weights[:, t] = w / np.nansum(w)

    return weights, all_curves