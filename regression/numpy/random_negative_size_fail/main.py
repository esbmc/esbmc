import numpy as np

try:
    np.random.normal(size=-1)
except ValueError:
    assert False
