import numpy as np

c = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
v = c[1]
assert v.sum() == 27
assert np.max(v) == 8
assert np.min(v) == 5
assert v.mean() == 6.5
assert v.any()
assert v.all()
