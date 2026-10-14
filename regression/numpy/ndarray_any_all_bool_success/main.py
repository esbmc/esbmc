import numpy as np

a = np.array([False, False, True, False])
b = np.array([True, True, True, True])

assert a.any() == True
assert a.all() == False
assert b.any() == True
assert b.all() == True
