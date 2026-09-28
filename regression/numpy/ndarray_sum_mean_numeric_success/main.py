import numpy as np

a = np.array([1, 2, 3, 4])
b = np.array([1.0, 2.0, 3.0])

assert a.sum() == 10
assert a.mean() == 2.5
assert b.sum() == 6.0
assert b.mean() == 2.0
