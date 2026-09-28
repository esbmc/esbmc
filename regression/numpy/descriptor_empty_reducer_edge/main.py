import numpy as np

a = np.array([1, 2, 3])
empty = a[1:1]

assert np.sum(empty) == 0
assert empty.any() == False
assert empty.all() == True

