import numpy as np

x = np.array([100, -100], dtype=np.int8)
u = np.array([200, 100], dtype=np.uint8)
mixed = x + u
assert mixed[0] == 44
