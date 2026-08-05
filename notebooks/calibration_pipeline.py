# %% [markdown]
# # NSM Model Calibration Pipeline
# Adaptation of Browning et al.'s (2024) online identification method
# to the uncontrolled stochastic NSM tumor growth model (Belkhatir et al., 2020).
#
# - Milestone 1: single-mouse calibration (simplified likelihood, sigma=0)
# - Milestone 2: population (8-mouse) joint calibration
# - Milestone 3: online/sequential update (get_weights)
#
# See src/calibration.py for the reusable functions; this notebook only
# runs them on the synthetic data and produces the results/figures.

# %% [markdown]
# ## Setup

# %%
import sys
sys.path.append("../src")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import emcee

from calibration import (
    log_posterior,
    log_posterior_population,
    get_weights_nsm,
)

# %%
data = pd.read_csv("../data/synthetic_NSM_tumor_data.csv")
print(data.head())

# %% [markdown]
# ## Milestone 1 — Single-mouse calibration
# Calibrate (a, b, alpha, V0) on mouse 1 alone, using the simplified
# likelihood (deterministic model, sigma=0, additive constant-variance
# measurement noise).

# %%
mouse1 = data[data.mouse_id == 1].sort_values("day")
print(mouse1)

true_a, true_b, true_V0 = mouse1.iloc[0][["a", "b", "V0"]]
print(f"\nVraies valeurs : a={true_a:.3f}, b={true_b:.3f}, V0={true_V0:.1f}")

# %%
observed_days = mouse1.day.values
observed_volumes = mouse1.V_obs.values

ndim = 4
nwalkers = 32

p0_center = np.array([1.0, 0.08, 0.6, 60.0])
p0 = p0_center + 1e-2 * p0_center * np.random.randn(nwalkers, ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior,
                                  args=(observed_days, observed_volumes))
sampler.run_mcmc(p0, 5000, progress=True)

# %%
samples = sampler.get_chain(discard=1000, thin=15, flat=True)

param_names = ["a", "b", "alpha", "V0"]
for i, name in enumerate(param_names):
    est = np.percentile(samples[:, i], [16, 50, 84])
    print(f"{name}: {est[1]:.3f}  (68% CI: [{est[0]:.3f}, {est[2]:.3f}])")

print(f"\nVraies valeurs : a={true_a:.3f}, b={true_b:.3f}, alpha=0.667, V0={true_V0:.1f}")

# %% [markdown]
# ### Diagnostic: is the found solution actually better, or is this identifiability?

# %%
true_params = [true_a, true_b, 2/3, true_V0]
found_params = list(np.percentile(samples, 50, axis=0))  # median MCMC estimate

print("log_posterior(vraies valeurs)   =", log_posterior(true_params, observed_days, observed_volumes))
print("log_posterior(valeurs trouvees) =", log_posterior(found_params, observed_days, observed_volumes))

# %%
from scipy.integrate import solve_ivp

def deterministic_curve(a, b, alpha, V0, days):
    def ode_rhs(t, V):
        return a * V**alpha - b * V
    sol = solve_ivp(ode_rhs, [0, max(days)], [V0], t_eval=days)
    return sol.y[0]

t_fine = np.linspace(0, 60, 200)
curve_true = deterministic_curve(true_a, true_b, 2/3, true_V0, t_fine)
curve_found = deterministic_curve(*found_params, t_fine)

plt.figure(figsize=(8, 5))
plt.plot(t_fine, curve_true, label="Courbe deterministe (vraies valeurs)", color="green")
plt.plot(t_fine, curve_found, label="Courbe deterministe (MCMC)", color="orange")
plt.scatter(observed_days, observed_volumes, color="black", zorder=5, label="Donnees observees (souris 1)")
plt.legend()
plt.xlabel("Jours"); plt.ylabel("Volume (mm3)")
plt.title("Milestone 1: deterministic-likelihood bias diagnostic")
plt.savefig("../results/milestone1_deterministic_bias.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# **Finding**: the true parameters no longer explain the data as well as
# the MCMC estimate, because the synthetic data include process noise
# (sigma=0.03) that this simplified likelihood (sigma=0) ignores. See
# docs/adaptation_plan.md for the full writeup.

# %% [markdown]
# ## Milestone 2 — Population (8-mouse) joint calibration
# a, b, alpha shared across all 8 mice (fixed effects, Eq. 20);
# each mouse keeps its own V0 (random effect, Eq. 21).

# %%
mice_days = []
mice_volumes = []
true_V0_list = []


for mid in range(1, 9):
    sub = data[data.mouse_id == mid].sort_values("day")
    mice_days.append(sub.day.values)
    mice_volumes.append(sub.V_obs.values)
    true_V0_list.append(sub.iloc[0]["V0"])

print("Vraies valeurs V0 par souris:", [f"{v:.1f}" for v in true_V0_list])

# %%
ndim_pop = 3 + 8   # a, b, alpha, V0_1...V0_8
nwalkers_pop = 64

p0_center_pop = np.array([1.0, 0.08, 0.6] + [50.0] * 8)
p0_pop = p0_center_pop + 1e-2 * np.abs(p0_center_pop) * np.random.randn(nwalkers_pop, ndim_pop)

sampler_pop = emcee.EnsembleSampler(nwalkers_pop, ndim_pop, log_posterior_population,
                                      args=(mice_days, mice_volumes))
sampler_pop.run_mcmc(p0_pop, 1000, progress=True)


# %%
samples_pop = sampler_pop.get_chain(discard=200, thin=5, flat=True)

param_names_pop = ["a", "b", "alpha"] + [f"V0_mouse{i}" for i in range(1, 9)]

print("=== Parametres partages (population) ===")
for i in range(3):
    est = np.percentile(samples_pop[:, i], [16, 50, 84])
    print(f"{param_names_pop[i]}: {est[1]:.3f}  (68% CI: [{est[0]:.3f}, {est[2]:.3f}])")

print("\n=== V0 par souris ===")
for i in range(3, 11):
    est = np.percentile(samples_pop[:, i], [16, 50, 84])
    true_v0 = true_V0_list[i - 3]
    print(f"{param_names_pop[i]}: {est[1]:.2f}  (68% CI: [{est[0]:.2f}, {est[2]:.2f}])  | vraie valeur: {true_v0:.2f}")

print("\nVraies valeurs partagees: a=1.300, b=0.090, alpha=0.667")

# %% [markdown]
# ## Milestone 3 — Online / sequential update
# Adapted from Browning et al.'s `get_weights()`. Uses the population
# samples (Milestone 2) as the prior, and re-weights them as mouse 1's
# measurements "arrive" one at a time.
#
# **Known limitation**: mouse 1 was part of the population fit above,
# so this is not yet a true out-of-sample prediction test.

# %%
idx_sub = np.random.choice(len(samples_pop), 500, replace=False)
param_samples = samples_pop[idx_sub, 0:3]   # a, b, alpha
V0_samples = samples_pop[idx_sub, 3]        # V0_mouse1 column, as a starting approximation

test_days = mice_days[0]
test_volumes = mice_volumes[0]

weights, curves = get_weights_nsm(param_samples, V0_samples, test_days, test_volumes)

print("Weights shape:", weights.shape)
print("Somme des poids a chaque temps (doit faire 1):", weights.sum(axis=0))

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
time_indices_to_show = [1, 4, 8]  # apres 2, 5, et 9 mesures recues

for ax, ti in zip(axes, time_indices_to_show):
    w = weights[:, ti]
    weighted_curve = np.average(curves, axis=0, weights=w)

    ax.plot(test_days, curves.T, color="gray", alpha=0.02)
    ax.plot(test_days, weighted_curve, color="red", linewidth=2, label="Prediction ponderee")
    ax.scatter(test_days[:ti + 1], test_volumes[:ti + 1], color="black", zorder=5, label="Donnees vues")
    ax.scatter(test_days[ti + 1:], test_volumes[ti + 1:], color="lightgray", zorder=5, label="Donnees futures (pas encore vues)")
    ax.set_title(f"Apres {ti + 1} mesures")
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig("../results/online_prediction_demo.png", dpi=150, bbox_inches="tight")
plt.show()



































############################ A F T E R ###################################


# %% [markdown]
# ## Milestone 4 — Monte Carlo likelihood (accounts for process noise)

# %%
from calibration import log_posterior_montecarlo

ndim_mc = 5   # a, b, alpha, sigma, V0
nwalkers_mc = 32

p0_center_mc = np.array([1.3, 0.09, 0.667, 0.03, 55.0])
p0_mc = p0_center_mc + 1e-2 * p0_center_mc * np.random.randn(nwalkers_mc, ndim_mc)

sampler_mc = emcee.EnsembleSampler(nwalkers_mc, ndim_mc, log_posterior_montecarlo,
                                     args=(observed_days, observed_volumes))
sampler_mc.run_mcmc(p0_mc, 10, progress=True)   # peu d'iterations d'abord, pour tester la vitesse

# %%
# Avec seulement 10 iterations, pas de vrai burn-in a faire -- on regarde juste
# la position actuelle des chaines pour voir si ca semble raisonnable
samples_mc = sampler_mc.get_chain(flat=True)

param_names_mc = ["a", "b", "alpha", "sigma", "V0"]
for i, name in enumerate(param_names_mc):
    est = np.percentile(samples_mc[:, i], [16, 50, 84])
    print(f"{name}: {est[1]:.3f}  (range vu: [{est[0]:.3f}, {est[2]:.3f}])")
    




# %%
sampler_mc2 = emcee.EnsembleSampler(nwalkers_mc, ndim_mc, log_posterior_montecarlo,
                                      args=(observed_days, observed_volumes))
sampler_mc2.run_mcmc(p0_mc, 10, progress=True)

# %%
# point de depart volontairement mauvais, pour un vrai test
p0_center_mc = np.array([0.5, 0.3, 0.4, 0.08, 20.0])   # tres different des vraies valeurs
p0_mc = p0_center_mc + 1e-2 * np.abs(p0_center_mc) * np.random.randn(nwalkers_mc, ndim_mc)

sampler_mc = emcee.EnsembleSampler(nwalkers_mc, ndim_mc, log_posterior_montecarlo,
                                     args=(observed_days, observed_volumes))
sampler_mc.run_mcmc(p0_mc, 1000, progress=True)




# %%

samples_mc = sampler_mc.get_chain(discard=100, thin=10, flat=True)

param_names_mc = ["a", "b", "alpha", "sigma", "V0"]
for i, name in enumerate(param_names_mc):
    est = np.percentile(samples_mc[:, i], [16, 50, 84])
    print(f"{name}: {est[1]:.3f}  (68% CI: [{est[0]:.3f}, {est[2]:.3f}])")

print(f"\nVraies valeurs : a=1.300, b=0.090, alpha=0.667, sigma=0.030, V0=55.0")
print(f"Point de depart (mauvais) : a=0.5, b=0.3, alpha=0.4, sigma=0.08, V0=20.0")


# %%

fig, axes = plt.subplots(5, 1, figsize=(10, 10), sharex=True)
chain = sampler_mc.get_chain()
for i, name in enumerate(param_names_mc):
    axes[i].plot(chain[:, :, i], alpha=0.3, color="black")
    axes[i].set_ylabel(name)
axes[-1].set_xlabel("Iteration")
plt.tight_layout()
plt.show()







































######################### A F T E R WITH 8 MICE ##########################

# %% [markdown]
# ## Milestone 5 — Monte Carlo likelihood, population (8 mice)
# Same idea as Milestone 4, but a, b, alpha, sigma are shared across
# all 8 mice (fixed effects), each mouse keeps its own V0.
# Starting point drawn from the prior (NOT the true values), to avoid
# the "cheating" issue we caught in Milestone 4.

# %%
from calibration import log_posterior_montecarlo_population

ndim_mcpop = 4 + 8   # a, b, alpha, sigma, V0_1...V0_8
nwalkers_mcpop = 24

# starting point drawn from the prior bounds, NOT from the true values
np.random.seed(123)  # for reproducibility of this particular start
p0_mcpop = np.column_stack([
    np.random.uniform(0.1, 5.0, nwalkers_mcpop),     # a
    np.random.uniform(0.01, 1.0, nwalkers_mcpop),    # b
    np.random.uniform(0.3, 0.99, nwalkers_mcpop),    # alpha
    np.random.uniform(0.001, 0.2, nwalkers_mcpop),   # sigma
] + [np.random.uniform(5.0, 200.0, nwalkers_mcpop) for _ in range(8)])




# %%
sampler_mcpop = emcee.EnsembleSampler(nwalkers_mcpop, ndim_mcpop,
                                        log_posterior_montecarlo_population,
                                        args=(mice_days, mice_volumes))

sampler_mcpop.run_mcmc(p0_mcpop, 5, progress=True)







# %%


samples_mcpop = sampler_mcpop.get_chain(discard=100, thin=5, flat=True)

param_names_mcpop = ["a", "b", "alpha", "sigma"] + [f"V0_mouse{i}" for i in range(1, 9)]

print("=== Parametres partages (population, Monte Carlo) ===")
for i in range(4):
    est = np.percentile(samples_mcpop[:, i], [16, 50, 84])
    print(f"{param_names_mcpop[i]}: {est[1]:.3f}  (68% CI: [{est[0]:.3f}, {est[2]:.3f}])")

print("\n=== V0 par souris ===")
for i in range(4, 12):
    est = np.percentile(samples_mcpop[:, i], [16, 50, 84])
    true_v0 = true_V0_list[i - 4]
    print(f"{param_names_mcpop[i]}: {est[1]:.2f}  (68% CI: [{est[0]:.2f}, {est[2]:.2f}])  | vraie valeur: {true_v0:.2f}")

print("\nVraies valeurs partagees: a=1.300, b=0.090, alpha=0.667, sigma=0.030")

# %%


sampler_mcpop = emcee.EnsembleSampler(nwalkers_mcpop, ndim_mcpop,
                                        log_posterior_montecarlo_population,
                                        args=(mice_days, mice_volumes))

import time
t0 = time.time()
sampler_mcpop.run_mcmc(p0_mcpop, 5, progress=True)
elapsed = time.time() - t0
print(f"\n{elapsed:.1f}s pour 5 iterations -> {elapsed/5:.2f}s/iteration")
print(f"Estimation pour 500 iterations: {elapsed/5*500/60:.1f} minutes")


# %%


sampler_mcpop.run_mcmc(None, 500, progress=True)


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

plt.suptitle("Milestone 5: shared parameters - estimate vs true (95% CI)")
plt.tight_layout()
plt.savefig("../results/milestone5_shared_params.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Plot 2: V0 per mouse ---
fig, ax = plt.subplots(figsize=(10, 5))
mouse_ids = list(range(1, 9))
medians = []
err_lows = []
err_highs = []
for i in range(4, 12):
    est = np.percentile(samples_mcpop[:, i], [2.5, 50, 97.5])
    medians.append(est[1])
    err_lows.append(est[1] - est[0])
    err_highs.append(est[2] - est[1])

ax.errorbar(mouse_ids, medians, yerr=[err_lows, err_highs],
            fmt='o', color='orange', markersize=8, capsize=6, label='Estimate (MCMC)')
ax.scatter(mouse_ids, true_V0_list, color='green', marker='*', s=200, zorder=5, label='True value')
ax.set_xlabel("Mouse")
ax.set_ylabel("V0 (mm3)")
ax.set_title("Milestone 5: V0 per mouse - estimate vs true (95% CI)")
ax.legend()
plt.tight_layout()
plt.savefig("../results/milestone5_v0_per_mouse.png", dpi=150, bbox_inches="tight")
plt.show()


# %%
