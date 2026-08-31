import numpy as np

a = np.array([0, 0, 3, 0])
b = np.array([1, 2, 3])
c = np.array([0.0, 0.0])

assert a.any() == True
assert a.all() == False
assert b.all() == True
assert c.any() == False
