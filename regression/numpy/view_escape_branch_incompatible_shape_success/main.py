import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
c = nondet_bool()
if c:
    v = a[0]
else:
    v = a[:, 0]
assert len(v) == 3 or len(v) == 2
