import numpy as np

a = np.array([1, 2, 3, 4])
part = a[1:3]
part[0] = 10

assert a[1] == 2
