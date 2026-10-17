import numpy as np
a = np.array([1, 2, 0, 4])
i = 0
while a[i] != 0 and i < 3:
    i += 1
assert i == 3
