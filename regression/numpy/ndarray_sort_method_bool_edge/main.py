import numpy as np

a = np.array([True, False, True, False])
a.sort()

assert a[0] == False
assert a[1] == False
assert a[2] == True
assert a[3] == True
