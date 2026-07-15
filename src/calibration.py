"""
calibration.py

Calibration / parameter identification for the uncontrolled stochastic
NSM model, for a single mouse (Milestone 1). Adapted from the logic of
inference.jl (Browning et al.), simplified to treat the NSM model as
deterministic (sigma=0) for this first version.
"""

import numpy as np
from nsm_model import simulate_one_mouse

def loglike(params, observed_days, observed_volumes):
    # TODO: on va l'écrire ensemble
    pass

def log_prior(params):
    # TODO
    pass

def log_posterior(params, observed_days, observed_volumes):
    # TODO
    pass
