import numpy as np

x = np.add(9223372036854775807, 1, dtype=np.int64)
assert x == -9223372036854775808
y = np.add(127, 1, dtype=np.int8)
assert y == -128
z = np.multiply(100, 3, dtype=np.uint8)
assert z == 44
