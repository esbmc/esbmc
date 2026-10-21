import numpy as np

try:
    np.random.choice([[1, 2], [3, 4]])
except ValueError:
    assert False
