import numpy as np
a = np.array([1, 2])
h = lambda: len(a)
if len(a) == 2:
    a = np.array([1, 2, 3])
assert h() == 2
