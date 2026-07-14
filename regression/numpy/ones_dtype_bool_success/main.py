import numpy as np

a = np.ones((1, 2), dtype=bool)
assert a[0][0] == True
assert a[0][1] == True
