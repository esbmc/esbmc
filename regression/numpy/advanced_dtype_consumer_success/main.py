import numpy as np

a = np.linspace(0, 3, 4, dtype=int)
assert np.sum(a) == 6
assert max(a) == 3
assert a.mean() == 1.5

b = np.array([100, 100], dtype=np.int8)
assert np.sum(b) == 200
assert np.prod(b) == 10000
assert np.mean(b) == 100.0
assert sum(b) == -56
c = np.add(b, b)
assert c[0] == -56 and c.itemsize == 1
