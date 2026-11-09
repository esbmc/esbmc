import numpy as np

x = np.array([1.0, 8.0, -27.0])
q = np.zeros(3)
np.cbrt(x, where=~(x > 0.0) | (x > 5.0), out=q)
assert q[0] == 1.0
