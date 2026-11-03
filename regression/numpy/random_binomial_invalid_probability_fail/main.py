import numpy as np

try:
    np.random.binomial(5, 1.5)
except ValueError:
    assert False
