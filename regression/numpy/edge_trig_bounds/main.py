import numpy as np

x = np.sin(1000.0)
y = np.cos(1000.0)

assert x <= 1.0
assert x >= -1.0
assert y <= 1.0
assert y >= -1.0
