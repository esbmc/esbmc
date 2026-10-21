import numpy as np

try:
    np.random.geometric(0.0)
except ValueError:
    assert False
