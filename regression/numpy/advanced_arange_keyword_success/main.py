import numpy as np

a = np.arange(start=2, stop=11, step=3)
b = np.arange(stop=3)
assert len(a) == 3 and a[2] == 8
assert len(b) == 3 and b[0] == 0 and b[2] == 2
