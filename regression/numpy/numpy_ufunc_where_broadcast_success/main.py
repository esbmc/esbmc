import numpy as np

o = np.zeros((2, 2))
np.hypot(np.array([[3.0], [5.0]]), np.array([4.0, 12.0]), where=[True, False], out=o)
assert o[0][0] >= 4.0 and o[0][0] < 5.7
assert o[1][1] == 0.0
p = np.full(2, 9.0)
np.arctan2(1.0, 1.0, where=[False, True], out=p)
assert p[0] == 9.0
assert p[1] > 0.0 and p[1] < 1.58
