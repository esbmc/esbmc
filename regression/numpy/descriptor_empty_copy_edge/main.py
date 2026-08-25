import numpy as np

a = np.array([1, 2, 3])
empty = a[1:1]

c = np.copy(empty)
d = np.array(empty)

assert len(c) == 0
assert len(d) == 0

