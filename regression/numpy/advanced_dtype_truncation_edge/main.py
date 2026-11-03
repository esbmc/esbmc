import numpy as np

a = np.array([2.999, -2.999, 0.5], dtype=np.int32)
b = np.array([255.9], dtype=np.uint8)
assert a[0] == 2 and a[1] == -2 and a[2] == 0
assert b[0] == 255
