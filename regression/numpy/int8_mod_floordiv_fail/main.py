import numpy as np

a = np.add(7, 0, dtype=np.int8)
b = np.add(3, 0, dtype=np.int8)
assert a % b == 0
