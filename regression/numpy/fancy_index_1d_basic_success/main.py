import numpy as np

a = np.array([10, 20, 30, 40])

assert a[[0, 2]][0] == 10
assert a[[0, 2]][1] == 30
