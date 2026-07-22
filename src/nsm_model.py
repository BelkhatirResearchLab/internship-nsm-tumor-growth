# ============================================================
# Synthetic Data Generation - Stochastic NSM Tumor Growth Model
# ============================================================
# Reference: Belkhatir et al., "Stochastic Norton-Simon-Massague
# Tumor Growth Modeling: Controlled and Mixed-Effect Uncontrolled
# Analysis", IEEE Trans. Control Systems Technology, 2020.
#
# We simulate the UNCONTROLLED version of the model (no drug/therapy
# term), described by the following stochastic differential
# equation (SDE) of Ito type (Equation 2 in the paper):
#
#       dV(t) = (a * V(t)^alpha - b * V(t)) dt + sigma * V(t)^beta * dW(t)
#
# where:
#   - V(t)  : tumor volume at time t
#   - a     : growth (anabolism) rate constant
#   - b     : death (catabolism) rate constant
#   - alpha : power-law growth exponent (0 < alpha < 1), related to
#             the fractal dimension of the proliferative tissue
#   - sigma : amplitude of the stochastic noise (biological/dynamical
#             heterogeneity, intrinsic to the tumor growth process)
#   - beta  : exponent controlling how the noise scales with volume
#             (beta = 1 => multiplicative noise, proportional to V)
#   - W(t)  : standard Wiener process (Brownian motion)
#
# POPULATION VARIABILITY STRUCTURE (validated with Zehor):
# Per Eq. 21 of the paper, the initial condition V0 varies between
# mice (random effect eta_i). IN ADDITION, following discussion with
# Zehor, we also let a and b vary slightly between mice (+/-10%
# coefficient of variation) to represent additional biological
# heterogeneity between individual tumors. This is a deliberate
# modeling choice on top of the paper's strict formulation (which
# only varies V0), validated with Zehor for this dataset.
#
# GOAL OF THIS SCRIPT:
# Generate synthetic tumor volume measurements for a population of
# mice (default: 8), with one measurement every 7 days over a
# 2-month period, mimicking a typical in vivo mouse experiment
# (similar in spirit to the lung/breast mice datasets used in the
# paper, Fig. 1). Each mouse follows its own trajectory of the SDE
# above (dynamical/process noise), and each measurement additionally
# includes observation/measurement noise (the same tumor, measured
# imperfectly).
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)  # fixed seed => results are reproducible every time you run this script

# ---------------------------------------------------------------
# 1) MODEL PARAMETERS (population-level / "fixed effects")
# ---------------------------------------------------------------
# These are the parameters shared by the WHOLE population of mice,
# before per-mouse variability is applied (see Section 5 below).

a_pop = 1.3     # growth rate constant a [mm^3^(1-alpha) / day]
b_pop = 0.09    # death rate constant b [1 / day]
alpha = 2/3     # power-law exponent (2/3 = classical von Bertalanffy
                # value, corresponds to surface-limited growth)
beta  = 1.0     # noise exponent. beta=1 => noise amplitude scales
                # linearly with current volume (Case 1 of Theorem 1
                # in the paper), which guarantees V(t) stays positive
                # for ANY a,b,sigma >= 0 -> safest numerical choice.
sigma = 0.03    # amplitude of the DYNAMICAL (process) noise. This is
                # the noise inside the SDE itself, NOT the measurement
                # noise (see Section 3 below for that).

V0_mean = 50.0  # average initial tumor volume across mice [mm^3]
V0_sd   = 10.0  # standard deviation of the initial volume across
                # mice (this implements the random effect eta_i on
                # the initial condition, Eq. 21 of the paper)

# Interindividual variability on a and b (biological heterogeneity
# between tumors), on top of the paper's V0-only formulation --
# validated with Zehor for this dataset.
interindiv_var_ab = True
a_cv = 0.10   # coefficient of variation of a across mice (10%)
b_cv = 0.10   # coefficient of variation of b across mice (10%)

# ---------------------------------------------------------------
# 2) EXPERIMENT SETUP
# ---------------------------------------------------------------
n_mice        = 8    # number of mice in the synthetic cohort
duration_days = 60   # total duration of the experiment (~2 months)
measure_every = 7    # a new measurement is taken every 7 days
dt            = 0.01 # integration time step for the Euler-Maruyama
                      # scheme [days]. Must be small compared to
                      # measure_every for an accurate simulation of
                      # the continuous-time SDE.

# time points at which we "measure" the tumor (0, 7, 14, ..., 56 days)
measurement_times = np.arange(0, duration_days + 1, measure_every)
n_meas = len(measurement_times)

# ---------------------------------------------------------------
# 3) MEASUREMENT NOISE
# ---------------------------------------------------------------
# This is a SECOND, separate source of noise: it represents the
# imprecision of measuring the tumor volume itself (e.g. caliper
# measurement error), added only at the discrete measurement times,
# on top of the true simulated volume V_true.
#
#   "additive"        : y = V_true + epsilon        (Gaussian noise,
#                       constant variance S, as specified by Zehor --
#                       matches Eq. 20 of the paper)
#   "multiplicative"   : y = V_true * exp(epsilon)   (log-normal noise,
#                       always keeps y > 0; matches Eq. 26 in the
#                       paper, used there for the model fitting itself)
#
# Default kept as "additive" per Zehor's instruction; the option
# below is left available in case we need to switch back for testing.

noise_type = "additive"
meas_sigma = 5.0     # standard deviation of the measurement noise
                     # (S = meas_sigma^2 in the paper's notation, Eq. 20)

# ---------------------------------------------------------------
# 4) SDE SIMULATION (EULER-MARUYAMA SCHEME)
# ---------------------------------------------------------------

def simulate_one_mouse(a, b, alpha, beta, sigma, V0, duration, dt):
    """
    Simulate one realization of the uncontrolled stochastic NSM
    model for a single mouse, using the Euler-Maruyama discretization
    of the SDE:
        dV = (a*V^alpha - b*V) dt + sigma*V^beta dW

    At every small time step dt, we add:
      - a deterministic "drift" step  : (a*V^alpha - b*V) * dt
      - a random "diffusion" step     : sigma * V^beta * dW,
        where dW ~ Normal(0, dt) is a fresh independent Gaussian
        draw at each step (this is what makes the trajectory
        fluctuate continuously, unlike a smooth deterministic ODE).

    Returns the full fine-grained trajectory (t_grid, V), which we
    only use internally to extract measurements at the correct
    days; it is NOT what we plot for the final figure shown to Zehor.
    """
    n_steps = int(duration / dt)
    t_grid = np.linspace(0, duration, n_steps + 1)
    V = np.zeros(n_steps + 1)
    V[0] = V0
    for k in range(n_steps):
        drift = (a * V[k]**alpha - b * V[k]) * dt
        dW = np.random.normal(0, np.sqrt(dt))
        diffusion = sigma * V[k]**beta * dW
        # np.maximum with a tiny floor avoids numerical issues if a
        # step accidentally pushes V slightly below zero
        V[k+1] = max(V[k] + drift + diffusion, 1e-6)
    return t_grid, V

# ---------------------------------------------------------------
# 5) GENERATE THE FULL SYNTHETIC COHORT
# ---------------------------------------------------------------
records = []            # will become the final tidy DataFrame (one row per measurement)
true_trajectories = {}  # keeps the fine-grained trajectory of each mouse (for diagnostics)

for mouse_id in range(1, n_mice + 1):

    # --- random effect on the initial condition (Eq. 21 in the paper) ---
    V0_i = max(np.random.normal(V0_mean, V0_sd), 1.0)

    # --- interindividual variability on a and b (heterogeneity) ---
    if interindiv_var_ab:
        a_i = max(np.random.normal(a_pop, a_cv * a_pop), 1e-3)
        b_i = max(np.random.normal(b_pop, b_cv * b_pop), 1e-3)
    else:
        a_i, b_i = a_pop, b_pop

    # --- simulate this mouse's full (fine time-step) trajectory ---
    t_grid, V_true = simulate_one_mouse(a_i, b_i, alpha, beta, sigma, V0_i, duration_days, dt)
    true_trajectories[mouse_id] = (t_grid, V_true)

    # --- extract the true volume at each of the 7-day measurement times ---
    idx_meas = np.searchsorted(t_grid, measurement_times)
    V_at_meas = V_true[idx_meas]

    # --- add measurement noise on top of the true volume ---
    if noise_type == "multiplicative":
        eps = np.random.normal(0, meas_sigma, size=n_meas)
        V_obs = V_at_meas * np.exp(eps)          # y = V * exp(eps), log-normal, always V > 0
    else:
        # constant-variance (homoscedastic) Gaussian noise, matching
        # eps_ij ~ N(0, S) in Eq. 20 of the paper -- S = meas_sigma^2,
        # same absolute noise level regardless of the current volume
        eps = np.random.normal(0, meas_sigma, size=n_meas)
        V_obs = V_at_meas + eps
        V_obs = np.clip(V_obs, 1e-3, None)

    # --- store one row per (mouse, measurement day) ---
    for day, v_true, v_obs in zip(measurement_times, V_at_meas, V_obs):
        records.append({
            "mouse_id": mouse_id,
            "day": day,
            "a": a_i, "b": b_i, "V0": V0_i,
            "V_true": v_true,   # "ground truth" volume (only known because this is synthetic data)
            "V_obs": v_obs      # noisy volume, i.e. what a real experiment would actually measure
        })

df = pd.DataFrame(records)

# ---------------------------------------------------------------
# 6) SAVE AND PREVIEW THE DATASET
# ---------------------------------------------------------------
df.to_csv("synthetic_NSM_tumor_data.csv", index=False)
print(df.head(12))
print(f"\nTotal: {n_mice} mice x {n_meas} measurements = {len(df)} rows")

# ---------------------------------------------------------------
# 7) VISUALIZATION
# ---------------------------------------------------------------
# NOTE: we intentionally plot ONLY the measured points (connected by
# straight lines), not the fine-grained continuous trajectory. This
# matches how tumor growth data is shown in the literature (e.g. Fig. 1
# of the NSM paper): real experiments never observe the continuous
# path, only the discrete weekly measurements.

plt.figure(figsize=(9, 6))
colors = plt.cm.tab10(np.linspace(0, 1, n_mice))

for mouse_id in range(1, n_mice + 1):
    sub = df[df.mouse_id == mouse_id].sort_values("day")
    plt.plot(sub.day, sub.V_obs, marker='o', color=colors[mouse_id-1],
              linestyle='-', markersize=6, label=f"Mouse {mouse_id}")

plt.xlabel("Time (days)")
plt.ylabel("Tumor volume V(t) (mm^3)")
plt.title("Synthetic data - uncontrolled stochastic NSM model")
plt.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.show()