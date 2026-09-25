import numpy as np

base = np.array([1], dtype=int)
out = np.empty_like(base)

assert out[0] == 0
