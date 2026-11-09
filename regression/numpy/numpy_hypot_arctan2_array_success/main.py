import math
import numpy as np

x = nondet_float()
__ESBMC_assume(x >= 1.0 and x <= 2.0)
h = np.hypot(np.array([3.0, x]), np.array([4.0, 0.0]))
assert h[0] == 5.0 or (h[0] >= 4.0 and h[0] < 5.66)
assert h[1] == x
m = np.hypot(np.array([[3.0], [5.0]]), np.array([[4.0], [12.0]]))
assert m[1][0] >= 12.0 and m[1][0] < 17.0
s = np.hypot(np.array([3.0, 5.0]), 4.0)
assert s[0] >= 4.0 and s[1] >= 5.0
b = np.hypot(np.array([[1.0, 2.0], [3.0, 4.0]]), [0.0, 1.0])
assert b[0][0] == 1.0 and b[1][0] == 3.0 and b[1][1] >= 4.0
a = np.arctan2(1.0, np.array([1.0, -1.0, np.inf]))
assert a[0] >= 0.785 and a[0] <= 1.571
assert a[1] >= 1.570 and a[1] <= 2.357
assert a[2] == 0.0
v = np.array([1.0, 0.0, x, x])
w = np.arctan2(v[::2], v[1::2])
assert w[0] == math.pi / 2 and w[1] >= 0.785 and w[1] <= 1.571
