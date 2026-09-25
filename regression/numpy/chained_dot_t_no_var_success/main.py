import numpy as np

a = np.zeros((2, 3)).T

assert a.shape[0] == 3
assert a.shape[1] == 2
