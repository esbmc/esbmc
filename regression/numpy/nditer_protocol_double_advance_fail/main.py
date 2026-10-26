import numpy as np

a = np.array([5, 6, 7])
it = np.nditer(a)
for x in it:
    next(it)
