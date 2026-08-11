# %% [markdown]
#  # NSM calibration
# 
#  Deterministic baseline, Monte Carlo likelihood, EKF likelihood, and
# 
#  a population prior built by calibrating each mouse individually and
# 
#  pooling the posteriors with a kernel density estimate.
# 
# 
# 
#  Every MCMC run below uses an HDF5 backend, so the chain is written
# 
#  to disk as it goes -- a crash or closed laptop doesn't lose progress.
# 
#  Each section has a "reload from disk" cell to read back a finished
# 
#  (or interrupted) run without recomputing it.

# %% [markdown]
#  ## Setup

# %%
%load_ext autoreload
%autoreload 2

import sys
sys.path.append("../src")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import emcee
import time
from scipy.integrate import solve_ivp

from calibration import (
    log_posterior,
    log_posterior_montecarlo,
    build_population_prior_kde,
    get_weights_nsm,
)
from ekf_likelihood import log_posterior_ekf


# %%
data = pd.read_csv("../data/synthetic_NSM_tumor_data.csv")
n_mice_total = data.mouse_id.nunique()
print(f"Loaded data for {n_mice_total} mice.")
print(data.head())


# %%
mouse1 = data[data.mouse_id == 1].sort_values("day")
observed_days = mouse1.day.values
observed_volumes = mouse1.V_obs.values

true_a, true_b, true_V0 = mouse1.iloc[0][["a", "b", "V0"]]
print(f"Mouse 1 true values: a={true_a:.3f}, b={true_b:.3f}, alpha=0.667, V0={true_V0:.1f}")


# %%
mice_days = []
mice_volumes = []
true_V0_list = []

for mid in sorted(data.mouse_id.unique()):
    sub = data[data.mouse_id == mid].sort_values("day")
    mice_days.append(sub.day.values)
    mice_volumes.append(sub.V_obs.values)
    true_V0_list.append(sub.iloc[0]["V0"])

n_mice = len(mice_days)
print(f"Population data ready: {n_mice} mice.")
print("True V0 per mouse:", [f"{v:.1f}" for v in true_V0_list])


# %% [markdown]
#  ## Deterministic likelihood (baseline)
# 
#  Fit mouse 1 alone assuming sigma=0 -- ignores the process noise that
# 
#  actually generated the data. Kept to show the bias this causes.

# %%
ndim = 4  # a, b, alpha, V0
nwalkers = 32

p0_center = np.array([1.0, 0.08, 0.6, 60.0])
p0 = p0_center + 1e-2 * p0_center * np.random.randn(nwalkers, ndim)

backend1 = emcee.backends.HDFBackend("../results/chain_deterministic.h5")
backend1.reset(nwalkers, ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior,
                                  args=(observed_days, observed_volumes),
                                  backend=backend1)
sampler.run_mcmc(p0, 5000, progress=True)


# %% [markdown]
#  ### reload from disk

# %%
backend1 = emcee.backends.HDFBackend("../results/chain_deterministic.h5")
sampler = backend1


# %%
samples = sampler.get_chain(discard=1000, thin=15, flat=True)

param_names = ["a", "b", "alpha", "V0"]
for i, name in enumerate(param_names):
    est = np.percentile(samples[:, i], [2.5, 50, 97.5])
    print(f"{name}: {est[1]:.3f}  (95% CI: [{est[0]:.3f}, {est[2]:.3f}])")

print(f"\nTrue values: a={true_a:.3f}, b={true_b:.3f}, alpha=0.667, V0={true_V0:.1f}")


# %%
found_params = list(np.percentile(samples, 50, axis=0))

def deterministic_curve(a, b, alpha, V0, days):
    def ode_rhs(t, V):
        return a * V**alpha - b * V
    sol = solve_ivp(ode_rhs, [0, max(days)], [V0], t_eval=days)
    return sol.y[0]

t_fine = np.linspace(0, 60, 200)
curve_true = deterministic_curve(true_a, true_b, 2/3, true_V0, t_fine)
curve_found = deterministic_curve(*found_params, t_fine)

plt.figure(figsize=(8, 5))
plt.plot(t_fine, curve_true, label="Deterministic curve (true values)", color="green")
plt.plot(t_fine, curve_found, label="Deterministic curve (MCMC estimate)", color="orange")
plt.scatter(observed_days, observed_volumes, color="black", zorder=5, label="Observed data (mouse 1)")
plt.legend()
plt.xlabel("Time (days)"); plt.ylabel("Volume (mm3)")
plt.title("Deterministic likelihood: bias when process noise is ignored")
plt.savefig("../results/deterministic_bias.png", dpi=150, bbox_inches="tight")
plt.show()


# %% [markdown]
#  The true parameters explain the data worse than the MCMC estimate --
# 
#  the fit compensates for the ignored process noise by distorting the
# 
#  parameter values. Motivates the Monte Carlo likelihood below.

# %% [markdown]
#  ## Monte Carlo likelihood, single mouse
# 
#  Same mouse, but the likelihood now simulates several noisy
# 
#  trajectories per parameter set instead of one smooth curve. Starting
# 
#  point is deliberately far from the true values, to check the fit
# 
#  actually converges from the data rather than just sitting near a
# 
#  convenient start.

# %%
ndim_mc = 5  # a, b, alpha, sigma, V0
nwalkers_mc = 32

p0_center_mc = np.array([0.5, 0.3, 0.4, 0.08, 20.0])
p0_mc = p0_center_mc + 1e-2 * np.abs(p0_center_mc) * np.random.randn(nwalkers_mc, ndim_mc)

backend2 = emcee.backends.HDFBackend("../results/chain_mc_single_mouse.h5")
backend2.reset(nwalkers_mc, ndim_mc)

sampler_mc = emcee.EnsembleSampler(nwalkers_mc, ndim_mc, log_posterior_montecarlo,
                                     args=(observed_days, observed_volumes),
                                     backend=backend2)
sampler_mc.run_mcmc(p0_mc, 1000, progress=True)


# %% [markdown]
#  ### reload from disk

# %%
backend2 = emcee.backends.HDFBackend("../results/chain_mc_single_mouse.h5")
sampler_mc = backend2


# %%
samples_mc = sampler_mc.get_chain(discard=100, thin=10, flat=True)

param_names_mc = ["a", "b", "alpha", "sigma", "V0"]
for i, name in enumerate(param_names_mc):
    est = np.percentile(samples_mc[:, i], [2.5, 50, 97.5])
    print(f"{name}: {est[1]:.3f}  (95% CI: [{est[0]:.3f}, {est[2]:.3f}])")

print(f"\nTrue values: a=1.300, b=0.090, alpha=0.667, sigma=0.030, V0=55.0")
print(f"Starting point (bad, on purpose): a=0.5, b=0.3, alpha=0.4, sigma=0.08, V0=20.0")


# %%
fig, axes = plt.subplots(5, 1, figsize=(10, 10), sharex=True)
chain = sampler_mc.get_chain()
for i, name in enumerate(param_names_mc):
    axes[i].plot(chain[:, :, i], alpha=0.3, color="black")
    axes[i].set_ylabel(name)
axes[-1].set_xlabel("Iteration")
plt.suptitle("Monte Carlo likelihood: trace plot, single mouse")
plt.tight_layout()
plt.savefig("../results/mc_trace_plot.png", dpi=150, bbox_inches="tight")
plt.show()


# %% [markdown]
#  sigma and V0 converge close to their true values. a, b, alpha settle
# 
#  on a stable but biased plateau -- likely an identifiability issue
# 
#  (b and alpha can partly offset each other in the growth equation)
# 
#  rather than a convergence problem. Motivates pooling multiple mice.

# %% [markdown]
#  ## EKF likelihood, single mouse
# 
#  Same mouse and starting point as above, but using the Kalman filter
# 
#  likelihood instead of Monte Carlo -- should be much faster per
# 
#  evaluation since it doesn't simulate repeated trajectories.

# %%
ndim_ekf = 5
nwalkers_ekf = 32

p0_center_ekf = np.array([0.5, 0.3, 0.4, 0.08, 20.0])
p0_ekf = p0_center_ekf + 1e-2 * np.abs(p0_center_ekf) * np.random.randn(nwalkers_ekf, ndim_ekf)

backend_ekf = emcee.backends.HDFBackend("../results/chain_ekf_single_mouse.h5")
backend_ekf.reset(nwalkers_ekf, ndim_ekf)

sampler_ekf = emcee.EnsembleSampler(nwalkers_ekf, ndim_ekf, log_posterior_ekf,
                                      args=(observed_days, observed_volumes),
                                      backend=backend_ekf)
sampler_ekf.run_mcmc(p0_ekf, 1000, progress=True)


# %% [markdown]
#  ### reload from disk

# %%
backend_ekf = emcee.backends.HDFBackend("../results/chain_ekf_single_mouse.h5")
sampler_ekf = backend_ekf


# %%
samples_ekf = sampler_ekf.get_chain(discard=100, thin=5, flat=True)

param_names_ekf = ["a", "b", "alpha", "sigma", "V0"]
for i, name in enumerate(param_names_ekf):
    est = np.percentile(samples_ekf[:, i], [2.5, 50, 97.5])
    print(f"{name}: {est[1]:.3f}  (95% CI: [{est[0]:.3f}, {est[2]:.3f}])")

print(f"\nTrue values: a=1.300, b=0.090, alpha=0.667, sigma=0.030, V0=55.0")


# %%
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
for i, (name, true_val) in enumerate(zip(param_names_ekf, [1.300, 0.090, 0.667, 0.030, 55.0])):
    est_mc = np.percentile(samples_mc[:, i], [2.5, 50, 97.5])
    est_ekf = np.percentile(samples_ekf[:, i], [2.5, 50, 97.5])

    axes[i].errorbar([0], [est_mc[1]], yerr=[[est_mc[1]-est_mc[0]], [est_mc[2]-est_mc[1]]],
                      fmt='o', color='orange', markersize=10, capsize=8, label='Monte Carlo')
    axes[i].errorbar([1], [est_ekf[1]], yerr=[[est_ekf[1]-est_ekf[0]], [est_ekf[2]-est_ekf[1]]],
                      fmt='o', color='purple', markersize=10, capsize=8, label='EKF')
    axes[i].scatter([0.5], [true_val], color='green', marker='*', s=250, zorder=5, label='True value')
    axes[i].set_title(name)
    axes[i].set_xticks([0, 1]); axes[i].set_xticklabels(['MC', 'EKF'])
    axes[i].legend(fontsize=7)

plt.suptitle("Monte Carlo vs EKF likelihood, single mouse")
plt.tight_layout()
plt.savefig("../results/mc_vs_ekf_comparison.png", dpi=150, bbox_inches="tight")
plt.show()


# %% [markdown]
#  ## Population prior: individual calibration + KDE pooling
# 
#  Calibrate each mouse on its own (reusing the EKF likelihood
# 
#  above), then pool the resulting samples with a kernel density
# 
#  estimate to get a population-level prior. Testing on a small subset
# 
#  first (3 mice) before scaling up to the full cohort.

# %%
ndim_ekf = 5
nwalkers_ekf = 32

mouse_subset = list(range(n_mice))  # les 15 souris
individual_posteriors_ekf = []

for idx in mouse_subset:
    days = mice_days[idx]
    vols = mice_volumes[idx]

    p0_center_i = np.array([1.0, 0.08, 0.6, 0.03, 55.0])
    p0_i = p0_center_i + 1e-2 * p0_center_i * np.random.randn(nwalkers_ekf, ndim_ekf)

    backend_i = emcee.backends.HDFBackend(f"../results/chain_individual_ekf_mouse{idx+1}.h5")
    backend_i.reset(nwalkers_ekf, ndim_ekf)

    sampler_i = emcee.EnsembleSampler(nwalkers_ekf, ndim_ekf, log_posterior_ekf,
                                        args=(days, vols), backend=backend_i)

    t0 = time.time()
    sampler_i.run_mcmc(p0_i, 400, progress=True)
    elapsed = time.time() - t0

    samples_i = sampler_i.get_chain(discard=80, thin=5, flat=True)
    individual_posteriors_ekf.append(samples_i[:, :4])  # a, b, alpha, sigma

    print(f"mouse {idx+1} done in {elapsed:.1f}s")

# %%
individual_posteriors_ekf = []

for idx in range(n_mice):
    backend_i = emcee.backends.HDFBackend(f"../results/chain_individual_ekf_mouse{idx+1}.h5")
    samples_i = backend_i.get_chain(discard=80, thin=5, flat=True)
    individual_posteriors_ekf.append(samples_i[:, :4])  # a, b, alpha, sigma

print(f"Loaded posteriors for {len(individual_posteriors_ekf)} mice.")

# %%
pooled_prior_ekf = build_population_prior_kde(individual_posteriors_ekf)

pooled_names = ["a", "b", "alpha", "sigma"]
for i, name in enumerate(pooled_names):
    est = np.percentile(pooled_prior_ekf[:, i], [2.5, 50, 97.5])
    print(f"{name} (pooled, EKF): {est[1]:.3f}  (95% CI: [{est[0]:.3f}, {est[2]:.3f}])")

print(f"\nTrue values: a=1.300, b=0.090, alpha=0.667, sigma=0.030")

# %%
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
true_values_pooled = [1.300, 0.090, 0.667, 0.030]

for i, (name, true_val) in enumerate(zip(pooled_names, true_values_pooled)):
    est = np.percentile(pooled_prior_ekf[:, i], [2.5, 50, 97.5])
    axes[i].errorbar([0], [est[1]], yerr=[[est[1]-est[0]], [est[2]-est[1]]],
                      fmt='o', color='purple', markersize=10, capsize=8, label='Pooled (EKF)')
    axes[i].scatter([0], [true_val], color='green', marker='*', s=250, zorder=5, label='True value')
    axes[i].set_title(name)
    axes[i].set_xticks([])
    axes[i].legend(fontsize=8)

plt.suptitle("Population prior via individual calibration + KDE pooling (15 mice, EKF)")
plt.tight_layout()
plt.savefig("../results/pooled_prior_ekf_15mice.png", dpi=150, bbox_inches="tight")
plt.show()


# %%



