import numpy as np

x = np.sin(-1e6)
assert abs(x) <= 1.0

y = np.sin(0.0) - 0.0
assert abs(y) < 1e-9

