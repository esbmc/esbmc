import numpy as np

a = np.full(0, 7)
b = np.full((0,), "x")
assert len(a) == 0
assert len(b) == 0
