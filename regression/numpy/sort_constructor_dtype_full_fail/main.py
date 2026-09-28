import numpy as np

a = np.sort(np.full((3,), 5.7, dtype=int))

assert a[0] == 6
