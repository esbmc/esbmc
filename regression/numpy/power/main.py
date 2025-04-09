import numpy as np

x = np.power(2, 3)
assert x == 8

y = np.power(2,7, dtype=np.int8)
assert y == -128
