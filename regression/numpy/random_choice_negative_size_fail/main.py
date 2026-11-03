import numpy as np

try:
    np.random.choice([1, 2], size=-1)
except ValueError:
    assert False
