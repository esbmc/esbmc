import numpy as np

a = np.array([1, 2])
b = np.array([3, 4])
for x, y, z in np.nditer([a, b]):
    pass
