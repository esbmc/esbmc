import numpy as np
x = np.power(2, 7, dtype=np.int8)
y = np.power(2, 7, dtype=np.uint8)
assert(x == y)
