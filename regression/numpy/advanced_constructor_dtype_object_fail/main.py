import numpy as np

a = np.full((2, ), 1, dtype="O")
assert a[1] == 1
