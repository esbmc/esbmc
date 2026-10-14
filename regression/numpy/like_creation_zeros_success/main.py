import numpy as np

base = np.array([[1, 2], [3, 4]], dtype=int)
out = np.zeros_like(base)

assert out[0][0] == 0
assert out[1][1] == 0
