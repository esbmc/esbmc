import numpy as np

a = np.arange(start=0, stop=8, step=2, dtype=float)
v = a[::-1]
assert v[0] == 6.0 and v[3] == 0.0
