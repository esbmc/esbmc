import numpy as np

try:
    np.random.beta(0.0, 1.0)
except ValueError:
    assert False
