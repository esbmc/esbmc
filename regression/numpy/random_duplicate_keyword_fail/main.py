import numpy as np

try:
    np.random.normal(0.0, loc=1.0)
except TypeError:
    assert False
