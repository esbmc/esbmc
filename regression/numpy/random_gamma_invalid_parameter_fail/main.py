import numpy as np

try:
    np.random.gamma(-1.0)
except ValueError:
    assert False
