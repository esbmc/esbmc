import numpy as np

try:
    np.random.default_rng(0)
except TypeError:
    assert False
