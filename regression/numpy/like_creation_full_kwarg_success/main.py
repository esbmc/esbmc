import numpy as np

base = np.ones((2,), dtype=int)
out = np.full_like(base, fill_value=7)

assert out[0] == 7
assert out[1] == 7
