import numpy as np

s = np.add(0.1, 0.2)
assert s != 0.3
assert s == 0.30000000000000004
q = np.divide(1.0, 3.0)
assert q * 3.0 == 1.0
