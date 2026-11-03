import numpy as np

a = np.random.random(2)
b = a.copy()
b[0] = 5.0

assert a[0] < 1.0
assert b[0] == 5.0
