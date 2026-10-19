import numpy as np

a = np.array([1, 2], dtype="float32")
b = np.zeros((2, 2), dtype=np.float16)
assert a[1] == 2.0
assert b[1][0] == 0.0
