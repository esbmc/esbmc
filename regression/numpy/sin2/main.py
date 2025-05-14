import numpy as np

x = np.sin(-1e6)
assert abs(x) <= 1.0
