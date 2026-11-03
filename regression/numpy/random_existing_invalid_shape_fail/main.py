import numpy as np

try:
    np.random.random(size=(2, -1))
except ValueError:
    assert False
