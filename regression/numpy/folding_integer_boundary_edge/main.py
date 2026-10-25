import numpy as np

x = np.add(9223372036854775806, 1, dtype=np.int64)
assert x == 9223372036854775807
y = np.add(126, 1, dtype=np.int8)
assert y == 127
