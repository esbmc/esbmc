import numpy as np

a = np.array([1, 2, 3, 4])
part = a[::-1]
a[0] = 99

assert part[3] == 99
