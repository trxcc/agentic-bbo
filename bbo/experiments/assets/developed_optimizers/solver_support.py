from bbo.core import *
from bbo.algorithms.model_based.gp_ei import GpEiAlgorithm

def create_reference_gp():
    return GpEiAlgorithm(**{'pool_size': None, 'startup_trials': 2, 'xi': 0.0, 'alpha': 1e-06, 'n_restarts_optimizer': 0, 'acquisition': 'ei', 'device': 'cpu'})
