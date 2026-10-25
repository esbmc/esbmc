import numpy as np

e = 2 ** np.array([0, 1, 3])
assert e[0] == 1 and e[2] == 8
p = np.array([2, 3]) ** np.array([3, 2])
assert p[0] == 8 and p[1] == 9
h = np.array([2.0, 4.0]) ** -1
assert h[0] == 0.5 and h[1] == 0.25
