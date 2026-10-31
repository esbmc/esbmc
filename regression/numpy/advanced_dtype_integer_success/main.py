import numpy as np

a = np.array([7, -3], dtype="int16")
b = np.ones(3, dtype="uint32")
assert a[0] == 7 and a[1] == -3
assert b[2] == 1

c = np.array([127, -128], dtype=np.int8)
d = np.zeros(2, dtype=np.uint8)
e = np.array([0.1, 2.5], dtype=np.float32)
c[0] += 1
d[1] = d[1] - 1
assert c[0] == -128
assert d[1] == 255
assert float(e[0]) != 0.1
assert e[1] == 2.5

assert c.dtype == np.int8
assert c.itemsize == 1 and c.nbytes == 2
assert e.itemsize == 4 and e.nbytes == 8
assert b.dtype == "uint32"
assert b.itemsize == 4 and b.nbytes == 12
