import numpy as np

base = np.ones((2, 1), dtype=int)
out = np.full_like(base, 7)

assert out[0][0] == 7
assert out[1][0] == 7
