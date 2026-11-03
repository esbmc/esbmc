import numpy as np

c = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
v = c[1]
assert len(v) == 2
assert v.shape == (2, 2)
assert v.ndim == 2
assert v.size == 4
