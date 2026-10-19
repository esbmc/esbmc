import numpy as np

a = np.arange(start=6, stop=0, step=-2)
assert len(a) == 3
assert a[0] == 6 and a[2] == 2
