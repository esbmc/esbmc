import numpy as np

z = np.zeros(0) + 1
assert len(z) == 0
w = np.zeros(0) + np.array([5])
assert len(w) == 0
v = np.array([5]) * np.zeros(0)
assert len(v) == 0
