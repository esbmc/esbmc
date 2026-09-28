import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
t1 = np.trace(a, offset=1)
t2 = np.trace(a, offset=-1)

assert t1 == 8
assert t2 == 12
