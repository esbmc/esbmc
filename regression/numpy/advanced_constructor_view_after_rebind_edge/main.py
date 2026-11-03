import numpy as np

a = np.arange(stop=4)
v = a[1:3]
a = np.zeros(4, dtype=int)
assert v[0] == 1 and v[1] == 2
assert len(a) == 4 and a[1] == 0
