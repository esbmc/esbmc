import numpy as np

try:
    np.random.poisson(-0.5, size=0)
except ValueError:
    assert False
