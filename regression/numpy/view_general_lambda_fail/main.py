import numpy as np

a = np.array([3, 1, 4, 1, 5])
v = a[1:4]
first_plus_len = lambda w: w[0] + len(w)
assert first_plus_len(v) == 3
