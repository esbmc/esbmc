import numpy as np

try:
    np.random.poisson(-1.0)
except ValueError:
    assert False
