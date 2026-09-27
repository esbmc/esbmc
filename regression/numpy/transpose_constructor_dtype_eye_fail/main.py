import numpy as np

base = np.full((2, 2), 4.9, dtype=int)
a = base.transpose()

assert a[0][0] == 5
