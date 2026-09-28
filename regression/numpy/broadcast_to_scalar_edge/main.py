import numpy as np

a = np.array([5])
v = np.broadcast_to(a, (2, 3))

assert v[0][0] == 5
assert v[0][2] == 5
assert v[1][1] == 5
