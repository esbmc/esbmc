import numpy as np

a = np.array([[7, 8, 9]])

assert len(a[0, :]) == 3
assert a[0, :][0] == 7
assert a[0, :][2] == 9
