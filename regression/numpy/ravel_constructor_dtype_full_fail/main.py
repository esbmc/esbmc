import numpy as np

a = np.ravel(np.full((2, 2), 3.5, dtype=int))

assert a[0] == 4
