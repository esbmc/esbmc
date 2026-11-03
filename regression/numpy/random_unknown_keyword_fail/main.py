import numpy as np

try:
    np.random.exponential(1.0, rate=2.0)
except TypeError:
    assert False
