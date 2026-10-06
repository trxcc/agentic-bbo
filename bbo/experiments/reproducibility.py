"""Isolate GP fit/pruning/acquisition RNG from earlier analysis calls."""
from contextlib import contextmanager, nullcontext
import hashlib
import random
import numpy as np


@contextmanager
def gp_rng(seed, history_size):
    try:
        import torch
    except ImportError:
        torch = None
    value = int(hashlib.sha256(f'gp_ei:{seed}:compute:{history_size}'.encode()).hexdigest()[:16], 16) % (2**31-1)
    py_state, np_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=[]) if torch is not None else nullcontext():
        try:
            random.seed(value)
            np.random.seed(value)
            if torch is not None:
                torch.manual_seed(value)
            yield
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
