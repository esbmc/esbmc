import numpy as np

a = np.arange(5, stop=8)
b = np.arange(5, 8)
assert len(a) == len(b) == 3
assert a[0] == b[0] == 5 and a[2] == 7
