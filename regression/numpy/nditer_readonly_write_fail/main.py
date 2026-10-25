import numpy as np

a = np.array([1, 2, 3])
for x in np.nditer(a):
    x[...] = 0
