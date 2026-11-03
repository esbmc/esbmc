import numpy as np

try:
    np.random.normal(0.0, -1.0)
except ValueError:
    assert False
