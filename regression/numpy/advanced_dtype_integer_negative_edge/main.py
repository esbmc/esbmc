import numpy as np

a = np.array([-128, 127], dtype=np.int8)
b = np.array([-0.9], dtype=np.uint8)
assert a[0] == -128 and a[1] == 127
assert b[0] == 0
