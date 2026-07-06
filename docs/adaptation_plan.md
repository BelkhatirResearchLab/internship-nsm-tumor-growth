# Adaptation plan: Browning et al. method -> NSM model (uncontrolled)

## Correspondence between the two frameworks

| Browning et al. (PSI model)              | NSM model (Belkhatir et al.)                     |
|-------------------------------------------|---------------------------------------------------|
| ODE forward model (deterministic)         | SDE forward model (stochastic): dV = (aV^a - bV)dt + sigma V^b dW |
| Parameters theta (growth/death rates)     | Parameters (a, b, alpha, sigma, beta)              |
| Gaussian observation noise on volume      | Multiplicative (log-normal) noise on volume        |
| First-level prior p1(theta)               | Prior on (a, b, alpha, sigma, beta) from literature/Table I |
| Per-patient posterior via MCMC/bootstrap  | Per-mouse posterior via MCMC on synthetic cohort   |
| Second-level prior p2(theta) (population) | Population prior built from the 8-mouse cohort     |
| Sequential update as weekly CT scans arrive | Sequential update as weekly volume measurements arrive |

## difference to handle
Browning's model is a deterministic ODE (all randomness is in the
observation noise). NSM is a stochastic SDE (randomness is also in
the dynamics itself). This means the likelihood function needs to
account for process noise, not just measurement noise -> the
forward model step must marginalize over SDE paths (or use a
particle-filter-style likelihood) rather than a single deterministic
trajectory.

## Next steps
- [ ] Implement NSM forward simulator compatible with a likelihood call (done: src/nsm_model.py)
- [ ] Define first-level prior on (a, b, alpha, sigma, beta)
- [ ] Implement per-mouse posterior sampling (MCMC, e.g. via emcee or PyMC)
- [ ] Build second-level (population) prior from the 8-mouse posteriors
- [ ] Implement sequential update as new weekly measurements arrive
