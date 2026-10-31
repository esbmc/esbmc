import numpy as np

a = np.full(3, (1, 2, 3))
assert a[2] == 3
b = np.full((2, 2), [1, 2])
assert b[0][0] == 1 and b[1][1] == 2
c = np.full((2, 2), [[1], [2]], dtype=np.int8)
assert c[0][1] == 1 and c[1][0] == 2 and c.itemsize == 1
c[1][0] += 127
c[1][0] += 127
assert c[1][0] == 0
d = np.full((2, 3), [1.5, 2.5, 3.5], dtype=np.float32)
assert d[1][2] == 3.5 and d.nbytes == 24
e = np.full(2, [1.5])
assert e[1] == 1.5
