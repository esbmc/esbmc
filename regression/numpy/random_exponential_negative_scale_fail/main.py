import numpy as np

try:
    np.random.exponential(-0.5)
except ValueError:
    assert False
