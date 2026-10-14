import numpy as np

base = np.zeros((2, 3), dtype=int)

assert base.shape[0] == 2
assert base.shape[1] == 3
assert base.ndim == 2
assert len(base) == 2
