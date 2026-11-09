import numpy as np

h = np.hypot(np.array([3.0, 5.0]), [4.0, 12.0])
assert h[0] > 4.999 and h[0] < 5.001
assert h[1] > 12.999 and h[1] < 13.001
g = np.hypot([[3.0], [6.0]], (4.0, 8.0))
assert g[1][1] > 9.999 and g[1][1] < 10.001
a = np.arctan2([1.0, -1.0], np.array([-1.0, -1.0]))
assert a[0] > 2.356 and a[0] < 2.357
assert a[1] < -2.356 and a[1] > -2.357
