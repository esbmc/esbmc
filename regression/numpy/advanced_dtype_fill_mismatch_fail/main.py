import numpy as np

a = np.array([127, -128], dtype=np.int8)
b = a + 1
assert b[0] == 128
