import numpy as np

a = np.full(2, -7.99, dtype=np.int16)
b = np.full(1, 255, dtype=np.uint8)
assert a[0] == -7 and a[1] == -7
assert b[0] == 255
