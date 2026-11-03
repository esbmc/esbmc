import numpy as np
from numpy import int16, float32

a = np.full(2, 7.9, dtype=int16)
b = np.arange(stop=2, dtype=float32)
assert a[0] == 7 and a[1] == 7
assert b[1] == 1.0
