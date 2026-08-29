import numpy as np

a = np.array([1, 2, 3])
v = np.broadcast_to(a, (2, 3))

assert len(v) == 2
assert v.shape[0] == 2
assert v.shape[1] == 3
assert v[0][0] == 1
assert v[0][2] == 3
assert v[1][1] == 2
