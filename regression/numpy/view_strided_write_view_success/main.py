import numpy as np

a = np.array([1, 2, 3, 4])
part = a[::2]
part[0] = 99

assert a[0] == 99
