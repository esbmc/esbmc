import numpy as np

try:
    np.random.chisquare(0)
except ValueError:
    assert False
