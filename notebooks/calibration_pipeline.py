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
#  ## Population prior: individual calibration + KDE pooling (EKF)
# 
#  Calibrate each mouse on its own (reusing the EKF likelihood
# 
#  above), then pool the resulting samples with a kernel density
# 
#  estimate to get a population-level prior. 

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

# %% [markdown]
#  ### reload from disk

# %%
individual_posteriors_ekf = []

for idx in range(n_mice):
    backend_i = emcee.backends.HDFBackend(f"../results/chain_individual_ekf_mouse{idx+1}.h5")
    samples_i = backend_i.get_chain(discard=80, thin=5, flat=True)
    individual_posteriors_ekf.append(samples_i[:, :5])  # a, b, alpha, sigma, V0 -- toutes les colonnes

print(f"Loaded posteriors for {len(individual_posteriors_ekf)} mice.")

# %%
np.random.seed(42)
pooled_prior_ekf = build_population_prior_kde(individual_posteriors_ekf)

pooled_names = ["a", "b", "alpha", "sigma", "V0"]
for i, name in enumerate(pooled_names):
    est = np.percentile(pooled_prior_ekf[:, i], [2.5, 50, 97.5])
    print(f"{name} (pooled, EKF): {est[1]:.3f}  (95% CI: [{est[0]:.3f}, {est[2]:.3f}])")

print(f"\nTrue values: a=1.300, b=0.090, alpha=0.667, sigma=0.030, V0~mean 50")

# %%
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
pooled_names = ["a", "b", "alpha", "sigma", "V0"]
true_values_pooled = [1.300, 0.090, 0.667, 0.030, 50.0]

for i, (name, true_val) in enumerate(zip(pooled_names, true_values_pooled)):
    est = np.percentile(pooled_prior_ekf[:, i], [2.5, 50, 97.5])
    axes[i].errorbar([0], [est[1]], yerr=[[est[1]-est[0]], [est[2]-est[1]]],
                      fmt='o', color='purple', markersize=10, capsize=8, label='Pooled (EKF)')
    axes[i].scatter([0], [true_val], color='green', marker='*', s=250, zorder=5, label='True value')
    axes[i].set_title(name)
    axes[i].set_xticks([])
    axes[i].legend(fontsize=8)
    axes[i].grid(True, alpha=1)

plt.suptitle("Population prior via individual calibration + KDE pooling (15 mice, EKF)")
plt.tight_layout()
plt.savefig("../results/pooled_prior_ekf_15mice.png", dpi=150, bbox_inches="tight")
plt.show()


# %% [markdown]
#  ## Population prior: individual calibration + KDE pooling (MC)
# 
#  Calibrate each mouse on its own (reusing the MC likelihood
# 
#  above), then pool the resulting samples with a kernel density
# 
#  estimate to get a population-level prior.

# %%
ndim_mc = 5
nwalkers_mc = 32

mouse_subset = list(range(n_mice))  # les 15 souris
individual_posteriors_mc = []

for idx in mouse_subset:
    days = mice_days[idx]
    vols = mice_volumes[idx]

    p0_center_i = np.array([1.0, 0.08, 0.6, 0.03, 55.0])
    p0_i = p0_center_i + 1e-2 * p0_center_i * np.random.randn(nwalkers_mc, ndim_mc)

    backend_i = emcee.backends.HDFBackend(f"../results/chain_individual_mc_mouse{idx+1}.h5")
    backend_i.reset(nwalkers_mc, ndim_mc)

    sampler_i = emcee.EnsembleSampler(nwalkers_mc, ndim_mc, log_posterior_montecarlo,
                                        args=(days, vols), backend=backend_i)

    t0 = time.time()
    sampler_i.run_mcmc(p0_i, 400, progress=True)
    elapsed = time.time() - t0

    samples_i = sampler_i.get_chain(discard=80, thin=5, flat=True)
    individual_posteriors_mc.append(samples_i[:, :4])  # a, b, alpha, sigma

    print(f"mouse {idx+1} done in {elapsed:.1f}s")

# %%
individual_posteriors_mc = []

for idx in range(n_mice):
    backend_i = emcee.backends.HDFBackend(f"../results/chain_individual_mc_mouse{idx+1}.h5")
    samples_i = backend_i.get_chain(discard=80, thin=5, flat=True)
    individual_posteriors_mc.append(samples_i[:, :5])


np.random.seed(42)
pooled_prior_mc = build_population_prior_kde(individual_posteriors_mc)

for i, name in enumerate(pooled_names):
    est = np.percentile(pooled_prior_mc[:, i], [2.5, 50, 97.5])
    print(f"{name} (pooled, MC): {est[1]:.3f}  (95% CI: [{est[0]:.3f}, {est[2]:.3f}])")

print(f"\nTrue values: a=1.300, b=0.090, alpha=0.667, sigma=0.030, V0~mean 50")

# %%
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
pooled_names = ["a", "b", "alpha", "sigma", "V0"]
true_values_pooled = [1.300, 0.090, 0.667, 0.030, 50.0]

for i, (name, true_val) in enumerate(zip(pooled_names, true_values_pooled)):
    est_ekf = np.percentile(pooled_prior_ekf[:, i], [2.5, 50, 97.5])
    est_mc = np.percentile(pooled_prior_mc[:, i], [2.5, 50, 97.5])

    axes[i].errorbar([0], [est_mc[1]], yerr=[[est_mc[1]-est_mc[0]], [est_mc[2]-est_mc[1]]],
                      fmt='o', color='orange', markersize=10, capsize=8, label='Monte Carlo')
    axes[i].errorbar([1], [est_ekf[1]], yerr=[[est_ekf[1]-est_ekf[0]], [est_ekf[2]-est_ekf[1]]],
                      fmt='o', color='purple', markersize=10, capsize=8, label='EKF')
    axes[i].scatter([0.5], [true_val], color='green', marker='*', s=250, zorder=5, label='True value')
    axes[i].set_title(name)
    axes[i].set_xticks([0, 1]); axes[i].set_xticklabels(['MC', 'EKF'])
    axes[i].legend(fontsize=7)
    axes[i].grid(True, alpha=1)

plt.suptitle("Pooled population prior: Monte Carlo vs EKF likelihood (15 mice)")
plt.tight_layout()
plt.savefig("../results/pooled_mc_vs_ekf_15mice.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ####################################################################################################################

# %% [markdown]
# ### Posteriors (EKF) - 15 mice

# %%
# %%
param_labels = ["a", "b", "alpha", "sigma", "V0"]
n_params = len(param_labels)
data = pooled_prior_ekf  # (2000, 5)

fig, axes = plt.subplots(n_params, n_params, figsize=(14, 14))

for i in range(n_params):
    for j in range(n_params):
        ax = axes[i, j]
        if i == j:
            ax.hist(data[:, i], bins=40, color="purple", alpha=0.7)
            ax.set_yticks([])
        elif i > j:
            ax.scatter(data[:, j], data[:, i], s=2, alpha=0.15, color="purple")
        else:
            ax.axis("off")

        if i == n_params - 1:
            ax.set_xlabel(param_labels[j])
        if j == 0 and i != 0:
            ax.set_ylabel(param_labels[i])

plt.suptitle("Pooled population prior (EKF, 15 mice) — pairwise distributions", y=1.02)
plt.tight_layout()
plt.savefig("../results/pooled_prior_cornerplot.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### Posteriors (MC) - 15 mice

# %%
# %%
param_labels = ["a", "b", "alpha", "sigma", "V0"]
n_params = len(param_labels)
data = pooled_prior_mc  # (2000, 5)

fig, axes = plt.subplots(n_params, n_params, figsize=(14, 14))

for i in range(n_params):
    for j in range(n_params):
        ax = axes[i, j]
        if i == j:
            ax.hist(data[:, i], bins=40, color="orange", alpha=0.7)
            ax.set_yticks([])
        elif i > j:
            ax.scatter(data[:, j], data[:, i], s=2, alpha=0.15, color="orange")
        else:
            ax.axis("off")

        if i == n_params - 1:
            ax.set_xlabel(param_labels[j])
        if j == 0 and i != 0:
            ax.set_ylabel(param_labels[i])

plt.suptitle("Pooled population prior (Monte Carlo, 15 mice) — pairwise distributions", y=1.02)
plt.tight_layout()
plt.savefig("../results/pooled_prior_cornerplot_mc.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ##########################################################################################################################################################################################
# ##########################################################################################################################################################################################
# ##########################################################################################################################################################################################
# ##########################################################################################################################################################################################
# ##########################################################################################################################################################################################
# ##########################################################################################################################################################################################
# ##########################################################################################################################################################################################
# ##########################################################################################################################################################################################

# %% [markdown]
# ##########################################################################################################################################################################################
# ##########################################################################################################################################################################################
# ##########################################################################################################################################################################################
# ##########################################################################################################################################################################################
# ##########################################################################################################################################################################################
# ##########################################################################################################################################################################################
# ##########################################################################################################################################################################################
# ##########################################################################################################################################################################################

# %% [markdown]
# ##########################################################################################################################################################################################
# ##########################################################################################################################################################################################
# ##########################################################################################################################################################################################
# ##########################################################################################################################################################################################
# ##########################################################################################################################################################################################
# ##########################################################################################################################################################################################
# ##########################################################################################################################################################################################
# ##########################################################################################################################################################################################

# %% [markdown]
# ##########################################################################################################################################################################################
# ##########################################################################################################################################################################################
# ##########################################################################################################################################################################################
# ##########################################################################################################################################################################################
# ##########################################################################################################################################################################################
# ##########################################################################################################################################################################################
# ##########################################################################################################################################################################################
# ##########################################################################################################################################################################################

# %% [markdown]
# ##########################################################################################################################################################################################
# ##########################################################################################################################################################################################
# ##########################################################################################################################################################################################
# ##########################################################################################################################################################################################
# ##########################################################################################################################################################################################
# ##########################################################################################################################################################################################
# ##########################################################################################################################################################################################
# ##########################################################################################################################################################################################

# %% [markdown]
# ##########################################################################################################################################################################################
# ##########################################################################################################################################################################################
# ##########################################################################################################################################################################################
# ##########################################################################################################################################################################################
# ##########################################################################################################################################################################################
# ##########################################################################################################################################################################################
# ##########################################################################################################################################################################################
# ##########################################################################################################################################################################################

# %% [markdown]
# ##### EXTRA JUST WANTED TO SEE THE EFFECT OF INITIAL POINTS ON THE ESTIMATION 

# %%
# %%
ndim_ekf = 5
nwalkers_ekf = 32

starting_points = {
    "far":   np.array([0.5, 0.3, 0.4, 0.08, 20.0]),
    "mid":   np.array([0.9, 0.19, 0.53, 0.055, 37.0]),
    "close": np.array([1.25, 0.10, 0.65, 0.032, 52.0]),
}

n_iterations = 1000
results = {}

for label, center in starting_points.items():
    p0 = center + 1e-2 * np.abs(center) * np.random.randn(nwalkers_ekf, ndim_ekf)

    backend_start = emcee.backends.HDFBackend(f"../results/chain_ekf_start_{label}.h5")
    backend_start.reset(nwalkers_ekf, ndim_ekf)

    sampler_start = emcee.EnsembleSampler(nwalkers_ekf, ndim_ekf, log_posterior_ekf,
                                            args=(observed_days, observed_volumes),
                                            backend=backend_start)

    t0 = time.time()
    sampler_start.run_mcmc(p0, n_iterations, progress=True)
    elapsed = time.time() - t0

    results[label] = sampler_start
    print(f"\n[{label}] done in {elapsed:.1f}s")

# %% [markdown]
# ### CELL 1
# 

# %%
# %%
colors = {"far": "red", "mid": "orange", "close": "green"}
param_names_ekf = ["a", "b", "alpha", "sigma", "V0"]

fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
for label, sampler_s in results.items():
    chain = sampler_s.get_chain()
    for i, name in enumerate(param_names_ekf):
        axes[i].plot(chain[:, :, i], alpha=0.15, color=colors[label])
        axes[i].plot([], [], color=colors[label], label=label)  # for legend
        axes[i].set_ylabel(name)

for ax in axes:
    ax.legend(fontsize=8, loc="upper right")
axes[-1].set_xlabel("Iteration")
plt.suptitle("EKF, single mouse: trace plots by starting point (far/mid/close)")
plt.tight_layout()
plt.savefig("../results/starting_point_trace_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### CELL 2

# %%
# %%
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
window = 50  # running median over this many iterations

for label, sampler_s in results.items():
    chain = sampler_s.get_chain()  # (n_iter, n_walkers, n_params)
    n_iter = chain.shape[0]
    for i, name in enumerate(param_names_ekf):
        running_median = [np.median(chain[max(0, k-window):k+1, :, i])
                           for k in range(n_iter)]
        axes[i].plot(running_median, color=colors[label], label=label)
        axes[i].set_title(name)

true_values_single = [1.300, 0.090, 0.667, 0.030, 55.0]
for i, tv in enumerate(true_values_single):
    axes[i].axhline(tv, color="black", linestyle="--", linewidth=1, label="true" if i == 0 else None)

axes[0].legend(fontsize=8)
plt.suptitle("Running median per parameter, by starting point -- shows when/where each converges")
plt.tight_layout()
plt.savefig("../results/starting_point_convergence_speed.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### CELL 3

# %%
# %%
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
true_values_single = [1.300, 0.090, 0.667, 0.030, 55.0]

for i, (name, true_val) in enumerate(zip(param_names_ekf, true_values_single)):
    for j, (label, sampler_s) in enumerate(results.items()):
        samples_s = sampler_s.get_chain(discard=200, thin=5, flat=True)
        est = np.percentile(samples_s[:, i], [2.5, 50, 97.5])
        axes[i].errorbar([j], [est[1]], yerr=[[est[1]-est[0]], [est[2]-est[1]]],
                          fmt='o', color=colors[label], markersize=10, capsize=8, label=label)
    axes[i].scatter([1], [true_val], color="black", marker="*", s=250, zorder=5, label="true")
    axes[i].set_title(name)
    axes[i].set_xticks(range(len(starting_points)))
    axes[i].set_xticklabels(list(starting_points.keys()))

axes[0].legend(fontsize=7)
plt.suptitle("Final estimate by starting point vs true value (EKF, single mouse, 1000 iterations)")
plt.tight_layout()
plt.savefig("../results/starting_point_final_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# also print numeric summary
for label, sampler_s in results.items():
    samples_s = sampler_s.get_chain(discard=200, thin=5, flat=True)
    print(f"\n--- {label} ---")
    for i, name in enumerate(param_names_ekf):
        est = np.percentile(samples_s[:, i], [2.5, 50, 97.5])
        print(f"  {name}: {est[1]:.3f}  (95% CI: [{est[0]:.3f}, {est[2]:.3f}])")

# %% [markdown]
# ### 200 iterations more to see if it is just because of needing more iterations or a structural problem

# %%
for label in ["far", "mid", "close"]:
    backend_s = emcee.backends.HDFBackend(f"../results/chain_ekf_start_{label}.h5")
    sampler_s = emcee.EnsembleSampler(nwalkers_ekf, ndim_ekf, log_posterior_ekf,
                                        args=(observed_days, observed_volumes),
                                        backend=backend_s)
    t0 = time.time()
    sampler_s.run_mcmc(None, 200, progress=True)  # juste 200 de plus, test rapide
    print(f"{label}: {time.time()-t0:.1f}s")
    results[label] = sampler_s

# %%
# %%
colors = {"far": "red", "mid": "orange", "close": "green"}
param_names_ekf = ["a", "b", "alpha", "sigma", "V0"]

fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
for label, sampler_s in results.items():
    chain = sampler_s.get_chain()
    for i, name in enumerate(param_names_ekf):
        axes[i].plot(chain[:, :, i], alpha=0.15, color=colors[label])
        axes[i].plot([], [], color=colors[label], label=label)  # for legend
        axes[i].set_ylabel(name)

for ax in axes:
    ax.legend(fontsize=8, loc="upper right")
axes[-1].set_xlabel("Iteration")
plt.suptitle("EKF, single mouse: trace plots by starting point (far/mid/close)")
plt.tight_layout()
plt.savefig("../results/starting_point_trace_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# %%
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
window = 50  # running median over this many iterations

for label, sampler_s in results.items():
    chain = sampler_s.get_chain()  # (n_iter, n_walkers, n_params)
    n_iter = chain.shape[0]
    for i, name in enumerate(param_names_ekf):
        running_median = [np.median(chain[max(0, k-window):k+1, :, i])
                           for k in range(n_iter)]
        axes[i].plot(running_median, color=colors[label], label=label)
        axes[i].set_title(name)

true_values_single = [1.300, 0.090, 0.667, 0.030, 55.0]
for i, tv in enumerate(true_values_single):
    axes[i].axhline(tv, color="black", linestyle="--", linewidth=1, label="true" if i == 0 else None)

axes[0].legend(fontsize=8)
plt.suptitle("Running median per parameter, by starting point -- shows when/where each converges")
plt.tight_layout()
plt.savefig("../results/starting_point_convergence_speed.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# %%
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
true_values_single = [1.300, 0.090, 0.667, 0.030, 55.0]

for i, (name, true_val) in enumerate(zip(param_names_ekf, true_values_single)):
    for j, (label, sampler_s) in enumerate(results.items()):
        samples_s = sampler_s.get_chain(discard=200, thin=5, flat=True)
        est = np.percentile(samples_s[:, i], [2.5, 50, 97.5])
        axes[i].errorbar([j], [est[1]], yerr=[[est[1]-est[0]], [est[2]-est[1]]],
                          fmt='o', color=colors[label], markersize=10, capsize=8, label=label)
    axes[i].scatter([1], [true_val], color="black", marker="*", s=250, zorder=5, label="true")
    axes[i].set_title(name)
    axes[i].set_xticks(range(len(starting_points)))
    axes[i].set_xticklabels(list(starting_points.keys()))

axes[0].legend(fontsize=7)
plt.suptitle("Final estimate by starting point vs true value (EKF, single mouse, 1000 iterations)")
plt.tight_layout()
plt.savefig("../results/starting_point_final_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# also print numeric summary
for label, sampler_s in results.items():
    samples_s = sampler_s.get_chain(discard=200, thin=5, flat=True)
    print(f"\n--- {label} ---")
    for i, name in enumerate(param_names_ekf):
        est = np.percentile(samples_s[:, i], [2.5, 50, 97.5])
        print(f"  {name}: {est[1]:.3f}  (95% CI: [{est[0]:.3f}, {est[2]:.3f}])")

# %% [markdown]
# #### ADDING 200 MORE ITERATIONS DOESN'T SOLVE ANYTHING

# %% [markdown]
# ### Now trying the 15, pooling, with a far initial point, still with EKF

# %%
# %%
ndim_ekf = 5
nwalkers_ekf = 32

far_start = np.array([0.5, 0.3, 0.4, 0.08, 20.0])

individual_posteriors_ekf_far = []

for idx in range(n_mice):
    days = mice_days[idx]
    vols = mice_volumes[idx]

    p0_i = far_start + 1e-2 * np.abs(far_start) * np.random.randn(nwalkers_ekf, ndim_ekf)

    backend_i = emcee.backends.HDFBackend(f"../results/chain_individual_ekf_far_mouse{idx+1}.h5")
    backend_i.reset(nwalkers_ekf, ndim_ekf)

    sampler_i = emcee.EnsembleSampler(nwalkers_ekf, ndim_ekf, log_posterior_ekf,
                                        args=(days, vols), backend=backend_i)

    t0 = time.time()
    sampler_i.run_mcmc(p0_i, 400, progress=True)
    elapsed = time.time() - t0

    samples_i = sampler_i.get_chain(discard=80, thin=5, flat=True)
    individual_posteriors_ekf_far.append(samples_i[:, :5])

    print(f"mouse {idx+1} done in {elapsed:.1f}s")

# %%
# %%
np.random.seed(42)
pooled_prior_ekf_far = build_population_prior_kde(individual_posteriors_ekf_far)

pooled_names_v0 = ["a", "b", "alpha", "sigma", "V0"]
for i, name in enumerate(pooled_names_v0):
    est = np.percentile(pooled_prior_ekf_far[:, i], [2.5, 50, 97.5])
    print(f"{name} (pooled, EKF, far start): {est[1]:.3f}  (95% CI: [{est[0]:.3f}, {est[2]:.3f}])")

print(f"\nTrue values: a=1.300, b=0.090, alpha=0.667, sigma=0.030, V0~50")

# %%
# %%
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
true_values_pooled = [1.300, 0.090, 0.667, 0.030, 50.0]

for i, (name, true_val) in enumerate(zip(pooled_names_v0, true_values_pooled)):
    est_close = np.percentile(pooled_prior_ekf_v0[:, i], [2.5, 50, 97.5])
    est_far = np.percentile(pooled_prior_ekf_far[:, i], [2.5, 50, 97.5])

    axes[i].errorbar([0], [est_close[1]], yerr=[[est_close[1]-est_close[0]], [est_close[2]-est_close[1]]],
                      fmt='o', color='green', markersize=10, capsize=8, label='Close start')
    axes[i].errorbar([1], [est_far[1]], yerr=[[est_far[1]-est_far[0]], [est_far[2]-est_far[1]]],
                      fmt='o', color='red', markersize=10, capsize=8, label='Far start')
    axes[i].scatter([0.5], [true_val], color='black', marker='*', s=250, zorder=5, label='True value')
    axes[i].set_title(name)
    axes[i].set_xticks([0, 1]); axes[i].set_xticklabels(['close', 'far'])
    axes[i].legend(fontsize=7)
    axes[i].grid(True, alpha=0.3)

plt.suptitle("Pooled prior: close vs far starting point (15 mice, EKF)")
plt.tight_layout()
plt.savefig("../results/pooled_close_vs_far_starting_point.png", dpi=150, bbox_inches="tight")
plt.show()


