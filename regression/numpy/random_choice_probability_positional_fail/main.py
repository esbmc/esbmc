import numpy as np

try:
    np.random.choice([1, 2], None, True, [0.5, 0.5])
except TypeError:
    assert False
