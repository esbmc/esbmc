import numpy as np

a = np.array([[3, 1], [2, 4]])
a.sort()  # Default axis=-1 sorts each row in place

assert a[0, 0] == 1
assert a[0, 1] == 3
assert a[1, 0] == 2
assert a[1, 1] == 4
