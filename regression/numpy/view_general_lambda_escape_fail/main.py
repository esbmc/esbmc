import numpy as np

a = np.array([3, 1, 4, 1, 5])
v = a[1:4]
keep = []
store = lambda w: keep.append(w)
store(v)
