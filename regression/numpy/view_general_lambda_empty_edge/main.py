import numpy as np

a = np.array([3, 1, 4, 1, 5])
v = a[1:4]
size = lambda w: len(w)
assert size(a[2:2]) == 0
