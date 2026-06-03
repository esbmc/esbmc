import numpy as np

x1 = np.exp(-20.0)
x2 = np.exp(-50.0)
assert x1 >= 0.0
assert x2 >= 0.0
assert x2 <= x1
assert x1 < 1.0
