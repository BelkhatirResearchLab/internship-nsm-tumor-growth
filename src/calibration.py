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





# THE ONE THAT DID'T WORKED, THIS FUNCTION COULD BE REMOVED
def get_weights_nsm_resampled(param_samples, V0_samples, observed_days, observed_volumes,
                                meas_sigma=5.0, ess_threshold_ratio=0.5):
    """
    Same idea as get_weights_nsm, but resamples the particle set
    whenever the effective sample size drops below a threshold
    (standard particle filter technique), to avoid weight degeneracy.
    """
    n_samples = len(param_samples)
    n_times = len(observed_days)
    ess_threshold = ess_threshold_ratio * n_samples

    current_params = param_samples.copy()
    current_V0 = V0_samples.copy()

    all_curves = np.full((n_samples, n_times), np.nan)
    weights_history = np.zeros((n_samples, n_times))
    log_w = np.zeros(n_samples)  # running log-weight since last resample

    for t_idx in range(n_times):
        day = observed_days[t_idx]
        y_obs = observed_volumes[t_idx]

        for i in range(n_samples):
            a, b, alpha = current_params[i]
            V0 = current_V0[i]

            def ode_rhs(t, V):
                return a * V**alpha - b * V

            sol = solve_ivp(ode_rhs, [0, day], [V0], method="RK45",
                             rtol=1e-4, atol=1e-4)
            if not sol.success:
                log_w[i] = -np.inf
                continue

            predicted = sol.y[0][-1]
            all_curves[i, t_idx] = predicted

            residual = y_obs - predicted
            log_w[i] += -0.5 * (residual / meas_sigma) ** 2

        # normalize weights
        finite_mask = np.isfinite(log_w)
        w = np.zeros(n_samples)
        w[finite_mask] = np.exp(log_w[finite_mask] - np.max(log_w[finite_mask]))
        w = w / w.sum()
        weights_history[:, t_idx] = w

        # resample if ESS too low
        ess = 1.0 / np.sum(w ** 2)
        if ess < ess_threshold and t_idx < n_times - 1:
            idx_resample = np.random.choice(n_samples, size=n_samples, p=w)
            current_params = current_params[idx_resample]
            current_V0 = current_V0[idx_resample]
            log_w = np.zeros(n_samples)  # reset weights after resampling

    return weights_history, all_curves


def get_weights_nsm_resampled_diversified(param_samples, V0_samples, observed_days, observed_volumes,
                                            meas_sigma=5.0, ess_threshold_ratio=0.5,
                                            diversify_scale=0.02):
    """
    Same as get_weights_nsm_resampled, but adds a small random
    perturbation after resampling (like a mini KDE kernel) to avoid
    particle impoverishment -- resampled copies stay slightly
    different from each other instead of becoming identical clones.
    """
    n_samples = len(param_samples)
    n_times = len(observed_days)
    ess_threshold = ess_threshold_ratio * n_samples

    current_params = param_samples.copy()
    current_V0 = V0_samples.copy()

    all_curves = np.full((n_samples, n_times), np.nan)
    weights_history = np.zeros((n_samples, n_times))
    log_w = np.zeros(n_samples)

    for t_idx in range(n_times):
        day = observed_days[t_idx]
        y_obs = observed_volumes[t_idx]

        for i in range(n_samples):
            a, b, alpha = current_params[i]
            V0 = current_V0[i]

            def ode_rhs(t, V):
                return a * V**alpha - b * V

            sol = solve_ivp(ode_rhs, [0, day], [V0], method="RK45",
                             rtol=1e-4, atol=1e-4)
            if not sol.success:
                log_w[i] = -np.inf
                continue

            predicted = sol.y[0][-1]
            all_curves[i, t_idx] = predicted

            residual = y_obs - predicted
            log_w[i] += -0.5 * (residual / meas_sigma) ** 2

        finite_mask = np.isfinite(log_w)
        w = np.zeros(n_samples)
        w[finite_mask] = np.exp(log_w[finite_mask] - np.max(log_w[finite_mask]))
        w = w / w.sum()
        weights_history[:, t_idx] = w

        ess = 1.0 / np.sum(w ** 2)
        if ess < ess_threshold and t_idx < n_times - 1:
            idx_resample = np.random.choice(n_samples, size=n_samples, p=w)

            # diversify: small perturbation scaled to each parameter's spread
            params_std = np.std(current_params, axis=0)
            V0_std = np.std(current_V0)

            noise_params = diversify_scale * params_std * np.random.randn(n_samples, 3)
            noise_V0 = diversify_scale * V0_std * np.random.randn(n_samples)

            current_params = current_params[idx_resample] + noise_params
            current_V0 = np.clip(current_V0[idx_resample] + noise_V0, 1.0, None)
            log_w = np.zeros(n_samples)

    return weights_history, all_curves, current_params





def get_weights_nsm_resampled_covariance(param_samples, V0_samples, observed_days, observed_volumes,
                                            meas_sigma=5.0, ess_threshold_ratio=0.5,
                                            diversify_scale=0.3):
    """
    Same as get_weights_nsm_resampled_diversified, but the
    post-resampling perturbation uses a Gaussian kernel with the
    FULL covariance matrix of the current particle set (like the KDE
    pooling step), instead of an isotropic gaussian per-dimension --
    this respects correlations between parameters (e.g. b/alpha)
    instead of perturbing each dimension independently.
    """
    n_samples = len(param_samples)
    n_times = len(observed_days)
    ess_threshold = ess_threshold_ratio * n_samples

    current_params = param_samples.copy()
    current_V0 = V0_samples.copy()

    all_curves = np.full((n_samples, n_times), np.nan)
    weights_history = np.zeros((n_samples, n_times))
    log_w = np.zeros(n_samples)

    for t_idx in range(n_times):
        day = observed_days[t_idx]
        y_obs = observed_volumes[t_idx]

        for i in range(n_samples):
            a, b, alpha = current_params[i]
            V0 = current_V0[i]

            def ode_rhs(t, V):
                return a * V**alpha - b * V

            sol = solve_ivp(ode_rhs, [0, day], [V0], method="RK45",
                             rtol=1e-4, atol=1e-4)
            if not sol.success:
                log_w[i] = -np.inf
                continue

            predicted = sol.y[0][-1]
            all_curves[i, t_idx] = predicted

            residual = y_obs - predicted
            log_w[i] += -0.5 * (residual / meas_sigma) ** 2

        finite_mask = np.isfinite(log_w)
        w = np.zeros(n_samples)
        w[finite_mask] = np.exp(log_w[finite_mask] - np.max(log_w[finite_mask]))
        w = w / w.sum()
        weights_history[:, t_idx] = w

        ess = 1.0 / np.sum(w ** 2)
        if ess < ess_threshold and t_idx < n_times - 1:
            idx_resample = np.random.choice(n_samples, size=n_samples, p=w)
            resampled_params = current_params[idx_resample]
            resampled_V0 = current_V0[idx_resample]

            # perturbation with the FULL covariance of the (pre-resample) params
            n_dim = 4  # a, b, alpha treated jointly + V0 separately
            combined = np.column_stack([current_params, current_V0])
            cov = np.cov(combined, rowvar=False)

            # Silverman-style bandwidth scaling on the covariance
            n_eff = n_samples
            bandwidth_scale = diversify_scale * (4 / (n_dim + 2))**(1/(n_dim+4)) * n_eff**(-1/(n_dim+4))
            perturbation = np.random.multivariate_normal(
                mean=np.zeros(n_dim), cov=cov * bandwidth_scale**2, size=n_samples
            )

            resampled_combined = np.column_stack([resampled_params, resampled_V0]) + perturbation
            current_params = resampled_combined[:, :3]
            current_V0 = np.clip(resampled_combined[:, 3], 1.0, None)
            log_w = np.zeros(n_samples)

    return weights_history, all_curves, current_params, current_V0




def get_weights_nsm_resampled_covariance_always(param_samples, V0_samples, observed_days, observed_volumes,
                                                    meas_sigma=5.0, diversify_scale=0.3):
    """
    Same as get_weights_nsm_resampled_covariance, but resamples and
    diversifies at EVERY measurement step, not just when ESS drops
    below a threshold -- prevents weight accumulation from ever
    building up enough to collapse.
    """
    n_samples = len(param_samples)
    n_times = len(observed_days)

    current_params = param_samples.copy()
    current_V0 = V0_samples.copy()

    all_curves = np.full((n_samples, n_times), np.nan)
    weights_history = np.zeros((n_samples, n_times))

    for t_idx in range(n_times):
        day = observed_days[t_idx]
        y_obs = observed_volumes[t_idx]

        log_w = np.zeros(n_samples)  # fresh weights each step, not cumulative

        for i in range(n_samples):
            a, b, alpha = current_params[i]
            V0 = current_V0[i]

            def ode_rhs(t, V):
                return a * V**alpha - b * V

            sol = solve_ivp(ode_rhs, [0, day], [V0], method="RK45",
                             rtol=1e-4, atol=1e-4)
            if not sol.success:
                log_w[i] = -np.inf
                continue

            predicted = sol.y[0][-1]
            all_curves[i, t_idx] = predicted

            residual = y_obs - predicted
            log_w[i] = -0.5 * (residual / meas_sigma) ** 2

        finite_mask = np.isfinite(log_w)
        w = np.zeros(n_samples)
        w[finite_mask] = np.exp(log_w[finite_mask] - np.max(log_w[finite_mask]))
        w = w / w.sum()
        weights_history[:, t_idx] = w

        if t_idx < n_times - 1:
            idx_resample = np.random.choice(n_samples, size=n_samples, p=w)
            resampled_params = current_params[idx_resample]
            resampled_V0 = current_V0[idx_resample]

            n_dim = 4
            combined = np.column_stack([current_params, current_V0])
            cov = np.cov(combined, rowvar=False)

            n_eff = n_samples
            bandwidth_scale = diversify_scale * (4 / (n_dim + 2))**(1/(n_dim+4)) * n_eff**(-1/(n_dim+4))
            perturbation = np.random.multivariate_normal(
                mean=np.zeros(n_dim), cov=cov * bandwidth_scale**2, size=n_samples
            )

            resampled_combined = np.column_stack([resampled_params, resampled_V0]) + perturbation
            current_params = resampled_combined[:, :3]
            current_V0 = np.clip(resampled_combined[:, 3], 1.0, None)

    return weights_history, all_curves, current_params, current_V0



def get_weights_nsm_particle_filter(param_samples, V0_samples, observed_days, observed_volumes,
                                      meas_sigma=5.0, diversify_scale=0.3, ess_threshold_ratio=0.5):
    """
    Proper particle filter structure: each particle's trajectory is
    simulated ONCE per segment (from the previous measurement day to
    the current one), continuing from where it left off -- not
    re-simulated from t=0 each time. Resampling/diversification only
    changes which particle continues forward, preserving trajectory
    continuity.
    """
    n_samples = len(param_samples)
    n_times = len(observed_days)
    ess_threshold = ess_threshold_ratio * n_samples

    current_params = param_samples.copy()   # (a, b, alpha) per particle, fixed for its lifetime unless resampled
    current_state = V0_samples.copy()       # current V(t) for each particle, evolves forward

    all_curves = np.full((n_samples, n_times), np.nan)
    weights_history = np.zeros((n_samples, n_times))
    log_w = np.zeros(n_samples)

    t_prev = 0.0

    for t_idx in range(n_times):
        day = observed_days[t_idx]
        y_obs = observed_volumes[t_idx]

        for i in range(n_samples):
            a, b, alpha = current_params[i]
            V_start = current_state[i]

            def ode_rhs(t, V):
                return a * V**alpha - b * V

            sol = solve_ivp(ode_rhs, [t_prev, day], [V_start], method="RK45",
                             rtol=1e-4, atol=1e-4)
            if not sol.success:
                log_w[i] = -np.inf
                current_state[i] = np.nan
                continue

            V_end = sol.y[0][-1]
            current_state[i] = V_end  # continue from here next segment
            all_curves[i, t_idx] = V_end

            residual = y_obs - V_end
            log_w[i] += -0.5 * (residual / meas_sigma) ** 2

        finite_mask = np.isfinite(log_w)
        w = np.zeros(n_samples)
        w[finite_mask] = np.exp(log_w[finite_mask] - np.max(log_w[finite_mask]))
        w = w / w.sum()
        weights_history[:, t_idx] = w

        ess = 1.0 / np.sum(w ** 2)
        if ess < ess_threshold and t_idx < n_times - 1:
            idx_resample = np.random.choice(n_samples, size=n_samples, p=w)
            current_params = current_params[idx_resample]
            current_state = current_state[idx_resample]  # continuity preserved: keeps the actual V(t) reached

            # diversify only the (a,b,alpha) going forward, not the state itself
            cov = np.cov(current_params, rowvar=False)
            bandwidth_scale = diversify_scale * (4 / 5)**(1/7) * n_samples**(-1/7)
            perturbation = np.random.multivariate_normal(
                mean=np.zeros(3), cov=cov * bandwidth_scale**2, size=n_samples
            )
            current_params = current_params + perturbation
            log_w = np.zeros(n_samples)

        t_prev = day

    return weights_history, all_curves