import numpy as np

a = np.array([5e-324])
s = np.sin(a)
assert s[0] >= 0.0 and s[0] < 1e-300
