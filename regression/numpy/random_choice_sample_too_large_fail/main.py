import numpy as np

try:
    np.random.choice([1, 2], size=3, replace=False)
except ValueError:
    assert False
