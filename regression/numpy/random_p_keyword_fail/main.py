import numpy as np

try:
    np.random.choice([1, 2, 3], size=2, p=[0.2, 0.3, 0.5])
except TypeError:
    assert False
