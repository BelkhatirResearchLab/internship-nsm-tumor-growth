# %% [markdown]
# # Cell of the SetUp

# %%
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

from calibration import build_population_prior_kde, get_weights_nsm

# %%
data = pd.read_csv("../data/synthetic_NSM_tumor_data.csv")

mice_days = []
mice_volumes = []
true_V0_list = []

for mid in sorted(data.mouse_id.unique()):
    sub = data[data.mouse_id == mid].sort_values("day")
    mice_days.append(sub.day.values)
    mice_volumes.append(sub.V_obs.values)
    true_V0_list.append(sub.iloc[0]["V0"])

n_mice = len(mice_days)
print(f"{n_mice} mice loaded.")

# %% [markdown]
# ## Reloading the runs that have already been made

# %%
# %%
individual_posteriors_ekf_v0 = []

for idx in range(n_mice):
    backend_i = emcee.backends.HDFBackend(f"../results/chain_individual_ekf_mouse{idx+1}.h5")
    samples_i = backend_i.get_chain(discard=80, thin=5, flat=True)
    individual_posteriors_ekf_v0.append(samples_i[:, :5])  # a, b, alpha, sigma, V0

print(f"Loaded {len(individual_posteriors_ekf_v0)} individual posteriors.")

# %% [markdown]
# ## Held-out test: pooled prior from close starting point

# %%
# %%
held_out_idx = 14  # mouse 15, kept aside as the "new" mouse

pool_for_prior = [individual_posteriors_ekf_v0[i] for i in range(n_mice) if i != held_out_idx]

np.random.seed(42)
pooled_prior_14 = build_population_prior_kde(pool_for_prior)

pooled_names_v0 = ["a", "b", "alpha", "sigma", "V0"]
for i, name in enumerate(pooled_names_v0):
    est = np.percentile(pooled_prior_14[:, i], [2.5, 50, 97.5])
    print(f"{name} (pooled, 14 mice, close): {est[1]:.3f}  (95% CI: [{est[0]:.3f}, {est[2]:.3f}])")

# %%
# %%
test_days = mice_days[held_out_idx]
test_volumes = mice_volumes[held_out_idx]

n_weight_samples = 500
idx_sub = np.random.choice(len(pooled_prior_14), n_weight_samples, replace=False)

param_samples = pooled_prior_14[idx_sub, 0:3]
V0_samples = pooled_prior_14[idx_sub, 4]

weights, curves = get_weights_nsm(param_samples, V0_samples, test_days, test_volumes)

max_reasonable_volume = 10000
valid_mask = np.all(curves < max_reasonable_volume, axis=1)
print(f"Keeping {valid_mask.sum()} / {len(valid_mask)} samples")

curves_clean = curves[valid_mask]
weights_clean = weights[valid_mask] / weights[valid_mask].sum(axis=0, keepdims=True)

# %% [markdown]
# ## Held-out test: pooled prior from far starting point

# %%
# %%
individual_posteriors_ekf_far = []

for idx in range(n_mice):
    backend_i = emcee.backends.HDFBackend(f"../results/chain_individual_ekf_far_mouse{idx+1}.h5")
    samples_i = backend_i.get_chain(discard=80, thin=5, flat=True)
    individual_posteriors_ekf_far.append(samples_i[:, :5])

pool_for_prior_far = [individual_posteriors_ekf_far[i] for i in range(n_mice) if i != held_out_idx]

np.random.seed(42)
pooled_prior_14_far = build_population_prior_kde(pool_for_prior_far)

for i, name in enumerate(pooled_names_v0):
    est = np.percentile(pooled_prior_14_far[:, i], [2.5, 50, 97.5])
    print(f"{name} (pooled, 14 mice, far): {est[1]:.3f}  (95% CI: [{est[0]:.3f}, {est[2]:.3f}])")

# %%
# %%
idx_sub_far = np.random.choice(len(pooled_prior_14_far), n_weight_samples, replace=False)

param_samples_far = pooled_prior_14_far[idx_sub_far, 0:3]
V0_samples_far = pooled_prior_14_far[idx_sub_far, 4]

weights_far, curves_far = get_weights_nsm(param_samples_far, V0_samples_far, test_days, test_volumes)

valid_mask_far = np.all(curves_far < max_reasonable_volume, axis=1)
print(f"Keeping {valid_mask_far.sum()} / {len(valid_mask_far)} samples")

curves_clean_far = curves_far[valid_mask_far]
weights_clean_far = weights_far[valid_mask_far] / weights_far[valid_mask_far].sum(axis=0, keepdims=True)

# %% [markdown]
# ## Combined figure: close vs far

# %%
# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True, sharex=True)
time_indices_to_show = [1, 4, 8]

for row, (curves_c, weights_c, row_label) in enumerate([
    (curves_clean, weights_clean, "Close start"),
    (curves_clean_far, weights_clean_far, "Far start"),
]):
    for col, ti in enumerate(time_indices_to_show):
        ax = axes[row, col]
        w = weights_c[:, ti]
        weighted_curve = np.average(curves_c, axis=0, weights=w)

        ax.plot(test_days, curves_c.T, color="gray", alpha=0.02)
        ax.plot(test_days, weighted_curve, color="red", linewidth=2, label="Weighted prediction")
        ax.scatter(test_days[:ti+1], test_volumes[:ti+1], color="black", zorder=5, label="Seen data")
        ax.scatter(test_days[ti+1:], test_volumes[ti+1:], color="lightgray", zorder=5, label="Future data")
        ax.set_ylim(0, 2500)
        if row == 0:
            ax.set_title(f"After {ti+1} measurements")
        if col == 0:
            ax.set_ylabel(row_label)
        if row == 0 and col == 0:
            ax.legend(fontsize=7)

plt.suptitle("Online update on held-out mouse 15: close vs far starting point")
plt.tight_layout()
plt.savefig("../results/online_update_heldout_close_vs_far.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# %%
def bayesian_r2(curves, weights_at_t, observed_days, observed_volumes, up_to_idx):
    """R2 bayesien (Browning et al., Eq. 10), sur les points DEJA vus jusqu'a up_to_idx."""
    r2_samples = []
    for s in range(curves.shape[0]):
        v_fit = curves[s, :up_to_idx+1]
        v_obs = observed_volumes[:up_to_idx+1]
        var_fit = np.var(v_fit)
        var_resid = np.var(v_fit - v_obs)
        if var_fit + var_resid > 0:
            r2_samples.append(var_fit / (var_fit + var_resid))
    r2_samples = np.array(r2_samples)
    w = weights_at_t[:len(r2_samples)]
    w = w / w.sum()
    # weighted median
    order = np.argsort(r2_samples)
    cum_w = np.cumsum(w[order])
    return r2_samples[order][np.searchsorted(cum_w, 0.5)]


def prediction_rmse(curves, weights_at_t, observed_days, observed_volumes, from_idx):
    """RMSE entre prediction ponderee et mesures FUTURES pas encore vues."""
    weighted_curve = np.average(curves, axis=0, weights=weights_at_t)
    future_pred = weighted_curve[from_idx+1:]
    future_obs = observed_volumes[from_idx+1:]
    if len(future_obs) == 0:
        return np.nan
    return np.sqrt(np.mean((future_pred - future_obs) ** 2))


def weighted_percentile(values, weights, percentile):
    order = np.argsort(values)
    values_sorted = values[order]
    weights_sorted = weights[order]
    cum_weights = np.cumsum(weights_sorted) / np.sum(weights_sorted)
    return values_sorted[np.searchsorted(cum_weights, percentile / 100)]


def coverage_95(curves, weights_at_t, observed_volumes, from_idx):
    """Fraction des mesures futures qui tombent dans l'IC 95% PONDERE des trajectoires."""
    future_obs = observed_volumes[from_idx+1:]
    if len(future_obs) == 0:
        return np.nan
    inside = []
    for j, day_idx in enumerate(range(from_idx+1, len(observed_volumes))):
        vals = curves[:, day_idx]
        lower = weighted_percentile(vals, weights_at_t, 2.5)
        upper = weighted_percentile(vals, weights_at_t, 97.5)
        inside.append(lower <= future_obs[j] <= upper)
    return np.mean(inside)


print("=== CLOSE start ===")
for ti in [1, 4, 8]:
    r2 = bayesian_r2(curves_clean, weights_clean[:, ti], test_days, test_volumes, ti)
    rmse = prediction_rmse(curves_clean, weights_clean[:, ti], test_days, test_volumes, ti)
    cov = coverage_95(curves_clean, weights_clean[:, ti], test_volumes, ti)
    print(f"After {ti+1} measurements: R2={r2:.3f}, RMSE(future)={rmse:.1f} mm3, coverage={cov:.2f}")

print("\n=== FAR start ===")
for ti in [1, 4, 8]:
    r2 = bayesian_r2(curves_clean_far, weights_clean_far[:, ti], test_days, test_volumes, ti)
    rmse = prediction_rmse(curves_clean_far, weights_clean_far[:, ti], test_days, test_volumes, ti)
    cov = coverage_95(curves_clean_far, weights_clean_far[:, ti], test_volumes, ti)
    print(f"After {ti+1} measurements: R2={r2:.3f}, RMSE(future)={rmse:.1f} mm3, coverage={cov:.2f}")

# %%
# %%
def effective_sample_size(weights_at_t):
    return 1.0 / np.sum(weights_at_t ** 2)

print("=== CLOSE start — effective sample size ===")
for ti in [1, 4, 8]:
    ess = effective_sample_size(weights_clean[:, ti])
    print(f"After {ti+1} measurements: ESS = {ess:.1f} / {len(weights_clean)} samples")

print("\n=== FAR start — effective sample size ===")
for ti in [1, 4, 8]:
    ess = effective_sample_size(weights_clean_far[:, ti])
    print(f"After {ti+1} measurements: ESS = {ess:.1f} / {len(weights_clean_far)} samples")

# %%
#### PASSER DE 500 A 5000 CAR RESULATS DEGENRESCENCE

# %%
# %%
np.random.seed(42)
pooled_prior_14 = build_population_prior_kde(pool_for_prior, n_output_samples=5000)

n_weight_samples = 5000
idx_sub = np.random.choice(len(pooled_prior_14), n_weight_samples, replace=False)

param_samples = pooled_prior_14[idx_sub, 0:3]
V0_samples = pooled_prior_14[idx_sub, 4]

weights, curves = get_weights_nsm(param_samples, V0_samples, test_days, test_volumes)

max_reasonable_volume = 10000
valid_mask = np.all(curves < max_reasonable_volume, axis=1)
print(f"Keeping {valid_mask.sum()} / {len(valid_mask)} samples")

curves_clean = curves[valid_mask]
weights_clean = weights[valid_mask] / weights[valid_mask].sum(axis=0, keepdims=True)

# %%
# %%
np.random.seed(42)
pooled_prior_14_far = build_population_prior_kde(pool_for_prior_far, n_output_samples=5000)

idx_sub_far = np.random.choice(len(pooled_prior_14_far), n_weight_samples, replace=False)

param_samples_far = pooled_prior_14_far[idx_sub_far, 0:3]
V0_samples_far = pooled_prior_14_far[idx_sub_far, 4]

weights_far, curves_far = get_weights_nsm(param_samples_far, V0_samples_far, test_days, test_volumes)

valid_mask_far = np.all(curves_far < max_reasonable_volume, axis=1)
print(f"Keeping {valid_mask_far.sum()} / {len(valid_mask_far)} samples")

curves_clean_far = curves_far[valid_mask_far]
weights_clean_far = weights_far[valid_mask_far] / weights_far[valid_mask_far].sum(axis=0, keepdims=True)

# %%
# %%
print("=== CLOSE start — effective sample size (5000 samples) ===")
for ti in [1, 4, 8]:
    ess = effective_sample_size(weights_clean[:, ti])
    print(f"After {ti+1} measurements: ESS = {ess:.1f} / {len(weights_clean)} samples")

print("\n=== FAR start — effective sample size (5000 samples) ===")
for ti in [1, 4, 8]:
    ess = effective_sample_size(weights_clean_far[:, ti])
    print(f"After {ti+1} measurements: ESS = {ess:.1f} / {len(weights_clean_far)} samples")

# %% [markdown]
# ## AFTER RESAMPLING

# %%
from calibration import get_weights_nsm_resampled

# %%
# %%
np.random.seed(42)
n_weight_samples = 500
idx_sub = np.random.choice(len(pooled_prior_14), n_weight_samples, replace=False)

param_samples = pooled_prior_14[idx_sub, 0:3]
V0_samples = pooled_prior_14[idx_sub, 4]

weights_resampled, curves_resampled = get_weights_nsm_resampled(
    param_samples, V0_samples, test_days, test_volumes
)

print("Weights shape:", weights_resampled.shape)
print("Sum of weights at each time:", weights_resampled.sum(axis=0))

# %%
# %%
print("=== CLOSE start — ESS with resampling ===")
for ti in [1, 4, 8]:
    ess = effective_sample_size(weights_resampled[:, ti])
    print(f"After {ti+1} measurements: ESS = {ess:.1f} / {n_weight_samples} samples")

# %%
# %%
print("Unique parameter combinations after resampling (should be << 500 if impoverished):")
unique_a = len(np.unique(np.round(param_samples[:, 0], 4)))
print(f"Unique 'a' values in current pool: {unique_a} / 500")

# %% [markdown]
# ### AFTER ADDING THE ONE WHICH IS DIVERSIFIED

# %%
from calibration import get_weights_nsm_resampled_diversified
# %%
np.random.seed(42)
weights_div, curves_div, final_params = get_weights_nsm_resampled_diversified(
    param_samples, V0_samples, test_days, test_volumes
)

print("=== CLOSE start — ESS with resampling + diversification ===")
for ti in [1, 4, 8]:
    ess = effective_sample_size(weights_div[:, ti])
    print(f"After {ti+1} measurements: ESS = {ess:.1f} / {n_weight_samples} samples")

print("\nUnique 'a' values in FINAL pool:", len(np.unique(np.round(final_params[:, 0], 4))), "/", n_weight_samples)

# %% [markdown]
# #### resampled covariance

# %%
from calibration import get_weights_nsm_resampled_covariance
np.random.seed(42)
weights_cov, curves_cov, final_params_cov, final_V0_cov = get_weights_nsm_resampled_covariance(
    param_samples, V0_samples, test_days, test_volumes
)

print("=== CLOSE start — ESS with covariance-aware resampling ===")
for ti in [1, 4, 8]:
    ess = effective_sample_size(weights_cov[:, ti])
    print(f"After {ti+1} measurements: ESS = {ess:.1f} / {n_weight_samples} samples")

print("\nUnique 'a' values in FINAL pool:", len(np.unique(np.round(final_params_cov[:, 0], 4))), "/", n_weight_samples)

# %% [markdown]
# #### resampled covariance always

# %%
from calibration import get_weights_nsm_resampled_covariance_always
# %%
np.random.seed(42)
weights_always, curves_always, final_params_always, final_V0_always = get_weights_nsm_resampled_covariance_always(
    param_samples, V0_samples, test_days, test_volumes
)

print("=== ESS with resampling at EVERY step ===")
for ti in [1, 4, 8]:
    ess = effective_sample_size(weights_always[:, ti])
    print(f"After {ti+1} measurements: ESS = {ess:.1f} / {n_weight_samples} samples")

print("\nUnique 'a' values in FINAL pool:", len(np.unique(np.round(final_params_always[:, 0], 4))), "/", n_weight_samples)

# %% [markdown]
# ####### Going from strict to not strict

# %%
# %%
np.random.seed(42)
weights_v2, curves_v2, final_params_v2, final_V0_v2 = get_weights_nsm_resampled_covariance_always(
    param_samples, V0_samples, test_days, test_volumes,
    meas_sigma=15.0,       # plus permissif, au lieu de 5.0
    diversify_scale=1.0    # plus de dispersion, au lieu de 0.3
)

print("=== ESS with wider meas_sigma + stronger diversification ===")
for ti in [1, 4, 8]:
    ess = effective_sample_size(weights_v2[:, ti])
    print(f"After {ti+1} measurements: ESS = {ess:.1f} / {n_weight_samples} samples")

print("\nUnique 'a' values in FINAL pool:", len(np.unique(np.round(final_params_v2[:, 0], 4))), "/", n_weight_samples)

# %% [markdown]
# ##### does it change prediction precision ?

# %%
# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
time_indices_to_show = [1, 4, 8]

for ax, ti in zip(axes, time_indices_to_show):
    w = weights_v2[:, ti]
    weighted_curve = np.average(curves_v2, axis=0, weights=w)

    ax.plot(test_days, curves_v2.T, color="gray", alpha=0.05)
    ax.plot(test_days, weighted_curve, color="red", linewidth=2, label="Weighted prediction")
    ax.scatter(test_days[:ti+1], test_volumes[:ti+1], color="black", zorder=5, label="Seen data")
    ax.scatter(test_days[ti+1:], test_volumes[ti+1:], color="lightgray", zorder=5, label="Future data")
    ax.set_ylim(0, 2500)
    ax.set_title(f"After {ti+1} measurements")
    ax.legend(fontsize=7)

plt.suptitle("Online update with covariance-aware resampling + wider likelihood (fixes weight degeneracy)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ##### FILTER TENTATIVE

# %%
# %%
from calibration import get_weights_nsm_particle_filter
np.random.seed(42)
weights_pf, curves_pf = get_weights_nsm_particle_filter(
    param_samples, V0_samples, test_days, test_volumes,
    meas_sigma=15.0, diversify_scale=1.0
)

print("=== Proper particle filter — ESS ===")
for ti in [1, 4, 8]:
    ess = effective_sample_size(weights_pf[:, ti])
    print(f"After {ti+1} measurements: ESS = {ess:.1f} / {n_weight_samples} samples")

# %%
# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True, sharex=True)
time_indices_to_show = [1, 4, 8]

for row, (curves_r, weights_r, row_label) in enumerate([
    (curves_clean, weights_clean, "Before (original method)"),
    (curves_pf, weights_pf, "After (particle filter fix)"),
]):
    for col, ti in enumerate(time_indices_to_show):
        ax = axes[row, col]
        w = weights_r[:, ti]
        weighted_curve = np.average(curves_r, axis=0, weights=w)

        ax.plot(test_days, curves_r.T, color="gray", alpha=0.05)
        ax.plot(test_days, weighted_curve, color="red", linewidth=2, marker='o')
        ax.scatter(test_days[:ti+1], test_volumes[:ti+1], color="black", zorder=5)
        ax.scatter(test_days[ti+1:], test_volumes[ti+1:], color="lightgray", zorder=5)
        ax.set_ylim(0, 2500)
        if row == 0:
            ax.set_title(f"After {ti+1} measurements")
        if col == 0:
            ax.set_ylabel(row_label)

plt.suptitle("Online update, held-out mouse 15 (close start): before vs after particle filter fix")
plt.tight_layout()
plt.savefig("../results/particle_filter_before_after.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
time_indices_to_show = [1, 4, 8]

for ax, ti in zip(axes, time_indices_to_show):
    w = weights_pf[:, ti]
    weighted_curve = np.average(curves_pf, axis=0, weights=w)

    ax.plot(test_days, curves_pf.T, color="gray", alpha=0.1)
    ax.plot(test_days, weighted_curve, color="red", linewidth=2, marker='o', label="Weighted prediction")
    ax.scatter(test_days[:ti+1], test_volumes[:ti+1], color="black", zorder=5, label="Seen data")
    ax.scatter(test_days[ti+1:], test_volumes[ti+1:], color="lightgray", zorder=5, label="Future data")
    ax.set_ylim(0, 2500)
    ax.set_title(f"After {ti+1} measurements")
    ax.legend(fontsize=7)

plt.suptitle("Proper particle filter: covariance-aware")

# %%
# %%
print("=== Proper particle filter — full metrics ===")
for ti in [1, 4, 8]:
    r2 = bayesian_r2(curves_pf, weights_pf[:, ti], test_days, test_volumes, ti)
    rmse = prediction_rmse(curves_pf, weights_pf[:, ti], test_days, test_volumes, ti)
    cov = coverage_95(curves_pf, weights_pf[:, ti], test_volumes, ti)
    ess = effective_sample_size(weights_pf[:, ti])
    print(f"After {ti+1} measurements: R2={r2:.3f}, RMSE(future)={rmse:.1f} mm3, coverage={cov:.2f}, ESS={ess:.1f}")

# %% [markdown]
# ##### juste voir


