import numpy as np

a = np.array([1, 2], dtype="float32")
b = np.zeros((2, 2), dtype=np.float16)
assert a[1] == 2.0
assert b[1][0] == 0.0

narrow_values = np.array([0.1, 0.2], dtype=np.float32)
assert narrow_values.mean() == 0.15000000596046448
short_values = np.zeros(1, dtype=np.int16)
assert short_values.dtype.name == "int16"
