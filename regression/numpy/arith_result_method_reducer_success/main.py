import numpy as np

a = np.array([1.0, 2.0])
b = a * 2.0
assert b.sum() == 6.0
assert b.max() == 4.0
assert np.sum(b) == 6.0
c = a - 1.0
assert c.min() == 0.0
