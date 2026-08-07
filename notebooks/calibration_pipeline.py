# %% [markdown]
# # NSM Model Calibration Pipeline
# Adaptation of Browning et al.'s (2024) online identification method
# to the uncontrolled stochastic NSM tumor growth model (Belkhatir et al., 2020).
#
# - Milestone 1: single-mouse calibration with a DETERMINISTIC likelihood
#   (sigma=0). Kept as a diagnostic: it shows that ignoring process noise
#   in the likelihood biases the parameter estimates (see plot below).
#   Not a final result -- just the motivation for Milestones 2-3.
# - Milestone 2: single-mouse calibration with a MONTE CARLO likelihood
#   that properly accounts for process noise (sigma > 0).
# - Milestone 3: population (n-mouse) joint calibration, Monte Carlo
#   likelihood, shared (a, b, alpha, sigma), individual V0 per mouse.
#
# See src/calibration.py for the reusable functions; this notebook only
# runs them on the synthetic data and produces the results/figures.
#
# NOTE ON CHECKPOINTING: every MCMC sampler below uses an HDF5 backend
# (emcee.backends.HDFBackend), saved under results/*.h5. This writes
# the chain to disk progressively DURING the run, not just at the end
# -- so a crash, a closed laptop, or a kernel restart does not lose
# the run in progress. To reload a finished (or interrupted) run
# without recomputing anything, see the "Reload from disk" cells below.

# %% [markdown]
# ## Setup
# Everything needed by every milestone is loaded here ONCE, so each
# milestone below can be run independently without re-running the
# others first (just run this Setup section first).

# %%
# autoreload: any change saved in src/calibration.py or src/nsm_model.py
# is picked up automatically -- no need to restart the kernel or
# manually reload the module after editing those files.
%load_ext autoreload
%autoreload 2

import sys
sys.path.append("../src")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import emcee
from scipy.integrate import solve_ivp

from calibration import (
    log_posterior,                        # Milestone 1: deterministic likelihood
    log_posterior_montecarlo,             # Milestone 2: single-mouse MC likelihood
    log_posterior_montecarlo_population,  # Milestone 3: population MC likelihood
    get_weights_nsm,                      # online/sequential update (not yet re-run with MC likelihood)
)

# %%
# Load the full synthetic dataset (generated in src/nsm_model.py)
data = pd.read_csv("../data/synthetic_NSM_tumor_data.csv")
n_mice_total = data.mouse_id.nunique()
print(f"Loaded data for {n_mice_total} mice.")
print(data.head())

# %%
# Single-mouse data (mouse 1), used by Milestone 1 and Milestone 2
mouse1 = data[data.mouse_id == 1].sort_values("day")
observed_days = mouse1.day.values
observed_volumes = mouse1.V_obs.values

true_a, true_b, true_V0 = mouse1.iloc[0][["a", "b", "V0"]]
print(f"Mouse 1 true values: a={true_a:.3f}, b={true_b:.3f}, alpha=0.667, V0={true_V0:.1f}")

# %%
# Population data (all mice), used by Milestone 3
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
# ## Milestone 1 — Diagnostic: deterministic likelihood bias
# Calibrate (a, b, alpha, V0) on mouse 1 alone, using a SIMPLIFIED
# likelihood that ignores process noise (sigma=0, only measurement
# noise). This is expected -- not surprising -- to give biased
# estimates, since the data were generated WITH process noise
# (sigma=0.03). Kept here as a diagnostic / motivation, not a result.

# %%
ndim = 4  # a, b, alpha, V0
nwalkers = 32

p0_center = np.array([1.0, 0.08, 0.6, 60.0])
p0 = p0_center + 1e-2 * p0_center * np.random.randn(nwalkers, ndim)

backend1 = emcee.backends.HDFBackend("../results/milestone1_chain.h5")
backend1.reset(nwalkers, ndim)  # start a fresh chain (remove this line to resume an existing one)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior,
                                  args=(observed_days, observed_volumes),
                                  backend=backend1)
sampler.run_mcmc(p0, 5000, progress=True)

# %% [markdown]
# ### Reload from disk (Milestone 1)
# Run this instead of the cell above if you already have a finished
# (or interrupted) chain saved on disk and don't want to recompute it.

# %%
backend1 = emcee.backends.HDFBackend("../results/milestone1_chain.h5")
sampler = backend1  # backend supports get_chain() directly, same as a sampler

# %%
samples = sampler.get_chain(discard=1000, thin=15, flat=True)

param_names = ["a", "b", "alpha", "V0"]
for i, name in enumerate(param_names):
    est = np.percentile(samples[:, i], [2.5, 50, 97.5])
    print(f"{name}: {est[1]:.3f}  (95% CI: [{est[0]:.3f}, {est[2]:.3f}])")

print(f"\nTrue values: a={true_a:.3f}, b={true_b:.3f}, alpha=0.667, V0={true_V0:.1f}")

# %%
# Diagnostic plot: true-parameter curve vs MCMC-found-parameter curve,
# against the observed (noisy) data points.
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
plt.title("Milestone 1: deterministic-likelihood bias diagnostic")
plt.savefig("../results/milestone1_deterministic_bias.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# **Finding**: the true parameters explain the data worse than the
# MCMC estimate, because the simplified likelihood (sigma=0) ignores
# the process noise actually present in the data (sigma=0.03). This
# is the expected consequence of a misspecified likelihood, and is
# the motivation for Milestones 2-3 below.

# %% [markdown]
# ## Milestone 2 — Single-mouse calibration, Monte Carlo likelihood
# Same mouse (mouse 1), but the likelihood now properly accounts for
# process noise: for each parameter set, several noisy trajectories
# are simulated (instead of one deterministic curve), and the observed
# data is compared to the resulting mean + spread. Starting point is
# deliberately far from the true values, to check that the MCMC
# actually converges from the data rather than just staying near a
# "lucky" starting point.

# %%
ndim_mc = 5  # a, b, alpha, sigma, V0
nwalkers_mc = 32

# deliberately bad starting point (not the true values) -- see note above
p0_center_mc = np.array([0.5, 0.3, 0.4, 0.08, 20.0])
p0_mc = p0_center_mc + 1e-2 * np.abs(p0_center_mc) * np.random.randn(nwalkers_mc, ndim_mc)

backend2 = emcee.backends.HDFBackend("../results/milestone2_chain.h5")
backend2.reset(nwalkers_mc, ndim_mc)  # start a fresh chain (remove this line to resume an existing one)

sampler_mc = emcee.EnsembleSampler(nwalkers_mc, ndim_mc, log_posterior_montecarlo,
                                     args=(observed_days, observed_volumes),
                                     backend=backend2)
sampler_mc.run_mcmc(p0_mc, 1000, progress=True)

# %% [markdown]
# ### Reload from disk (Milestone 2)

# %%
backend2 = emcee.backends.HDFBackend("../results/milestone2_chain.h5")
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
# Trace plot: check convergence over the iterations for each parameter
fig, axes = plt.subplots(5, 1, figsize=(10, 10), sharex=True)
chain = sampler_mc.get_chain()
for i, name in enumerate(param_names_mc):
    axes[i].plot(chain[:, :, i], alpha=0.3, color="black")
    axes[i].set_ylabel(name)
axes[-1].set_xlabel("Iteration")
plt.suptitle("Milestone 2: MCMC trace plot (single mouse, Monte Carlo likelihood)")
plt.tight_layout()
plt.savefig("../results/milestone2_trace_plot.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# **Finding**: sigma and V0 converge close to their true values.
# a, b, alpha stabilize on a stable but biased plateau -- likely a
# structural identifiability issue (b and alpha can partly compensate
# each other in the growth equation) rather than a convergence
# failure, consistent with identifiability limitations discussed in
# Belkhatir et al. This motivates pooling multiple mice (Milestone 3).

# %% [markdown]
# ## Milestone 3 — Population calibration, Monte Carlo likelihood
# a, b, alpha, sigma are shared across all mice (fixed effects); each
# mouse keeps its own V0 (random effect). Starting point drawn from
# the prior bounds (not the true values), same reasoning as Milestone 2.

# %%
ndim_mcpop = 4 + n_mice  # a, b, alpha, sigma, V0_1...V0_n
nwalkers_mcpop = 24

np.random.seed(123)  # reproducibility of this particular starting point
p0_mcpop = np.column_stack([
    np.random.uniform(0.1, 5.0, nwalkers_mcpop),    # a
    np.random.uniform(0.01, 1.0, nwalkers_mcpop),   # b
    np.random.uniform(0.3, 0.99, nwalkers_mcpop),   # alpha
    np.random.uniform(0.001, 0.2, nwalkers_mcpop),  # sigma
] + [np.random.uniform(5.0, 200.0, nwalkers_mcpop) for _ in range(n_mice)])

backend3 = emcee.backends.HDFBackend("../results/milestone3_chain.h5")
backend3.reset(nwalkers_mcpop, ndim_mcpop)  # start a fresh chain (remove this line to resume an existing one)

sampler_mcpop = emcee.EnsembleSampler(nwalkers_mcpop, ndim_mcpop,
                                        log_posterior_montecarlo_population,
                                        args=(mice_days, mice_volumes),
                                        backend=backend3)

# %%
# Quick timing test (5 iterations) before committing to a long run
import time
t0 = time.time()
sampler_mcpop.run_mcmc(p0_mcpop, 5, progress=True)
elapsed = time.time() - t0
print(f"\n{elapsed:.1f}s for 5 iterations -> {elapsed/5:.2f}s/iteration")
print(f"Estimated time for 500 iterations: {elapsed/5*500/60:.1f} minutes")

# %%
# Full run -- continues the same chain (do not re-create the sampler,
# or re-run the "reset" cell above, or you lose the 5 iterations done)
sampler_mcpop.run_mcmc(None, 500, progress=True)

# %% [markdown]
# ### Reload from disk (Milestone 3)
# Use this cell to pick up a run you started earlier (e.g. yesterday)
# without recomputing anything -- whether it finished or was interrupted.

# %%
backend3 = emcee.backends.HDFBackend("../results/milestone3_chain.h5")
sampler_mcpop = backend3
print(f"Chain currently has {backend3.iteration} iterations saved on disk.")

# %% [markdown]
# To ADD more iterations to a reloaded run (rather than just reading
# it), re-create a real EnsembleSampler pointing at the same backend,
# then call run_mcmc(None, ...):
# ```
# sampler_mcpop = emcee.EnsembleSampler(nwalkers_mcpop, ndim_mcpop,
#                                         log_posterior_montecarlo_population,
#                                         args=(mice_days, mice_volumes),
#                                         backend=backend3)
# sampler_mcpop.run_mcmc(None, 500, progress=True)  # adds 500 more iterations
# ```

# %%
samples_mcpop = sampler_mcpop.get_chain(discard=100, thin=5, flat=True)

# --- Plot 1: shared parameters (a, b, alpha, sigma) ---
param_names_shared = ["a", "b", "alpha", "sigma"]
true_values_shared = [1.300, 0.090, 0.667, 0.030]

fig, axes = plt.subplots(1, 4, figsize=(14, 4))
for i, (name, true_val) in enumerate(zip(param_names_shared, true_values_shared)):
    est = np.percentile(samples_mcpop[:, i], [2.5, 50, 97.5])
    median = est[1]
    err_low = median - est[0]
    err_high = est[2] - median

    axes[i].errorbar([0], [median], yerr=[[err_low], [err_high]],
                      fmt='o', color='orange', markersize=10, capsize=8, label='Estimate (MCMC)')
    axes[i].scatter([0], [true_val], color='green', marker='*', s=250, zorder=5, label='True value')
    axes[i].set_title(name)
    axes[i].set_xticks([])
    axes[i].legend(fontsize=8)

plt.suptitle(f"Milestone 3: shared parameters - estimate vs true (95% CI, n_mice={n_mice})")
plt.tight_layout()
plt.savefig("../results/milestone3_shared_params.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Plot 2: V0 per mouse ---
fig, ax = plt.subplots(figsize=(12, 5))
mouse_ids = list(range(1, n_mice + 1))
medians, err_lows, err_highs = [], [], []
for i in range(4, 4 + n_mice):
    est = np.percentile(samples_mcpop[:, i], [2.5, 50, 97.5])
    medians.append(est[1])
    err_lows.append(est[1] - est[0])
    err_highs.append(est[2] - est[1])

ax.errorbar(mouse_ids, medians, yerr=[err_lows, err_highs],
            fmt='o', color='orange', markersize=8, capsize=6, label='Estimate (MCMC)')
ax.scatter(mouse_ids, true_V0_list, color='green', marker='*', s=200, zorder=5, label='True value')
ax.set_xlabel("Mouse")
ax.set_ylabel("V0 (mm3)")
ax.set_title(f"Milestone 3: V0 per mouse - estimate vs true (95% CI, n_mice={n_mice})")
ax.legend()
plt.tight_layout()
plt.savefig("../results/milestone3_v0_per_mouse.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# **Next steps (not yet done)**:
# - Compare this n_mice result to a smaller population (e.g. n_mice=8),
#   to check whether pooling more mice improves identifiability of b/alpha.
# - Re-run the online/sequential update (get_weights_nsm) using this
#   Monte Carlo population posterior, instead of the deterministic one
#   used previously.
# - Sensitivity analysis requested by Zehor: vary a, b, sigma, number
#   of mice, and measurement frequency, and observe the effect on
#   estimation quality.