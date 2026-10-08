import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[7, 8, 9], [1, 1, 1]])
c = nondet_bool()
if c:
    v = a[0]
else:
    v = b[0]
assert v[0] == 1 or v[0] == 7
