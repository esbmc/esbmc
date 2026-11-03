import numpy as np

c = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
v = c[1]
t = v.tolist()
assert len(t) == 2
assert len(t[0]) == 2
assert t[1][0] == 7
assert t[0][1] == 6
