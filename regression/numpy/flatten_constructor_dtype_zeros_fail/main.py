import numpy as np

base = np.full((2, 2), 2.9, dtype=int)
a = base.flatten()

assert a[0] == 3
