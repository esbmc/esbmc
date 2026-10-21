import numpy as np

try:
    np.random.randint(0, 5, size=2, dtype=np.float64)
except TypeError:
    assert False
