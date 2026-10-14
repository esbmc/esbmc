import numpy as np
def pick(cond):
    a = np.array([1, 2, 3, 4, 5])
    if cond:
        b = a[0:3]
    b[1] = 0
    return a[1]
assert pick(True) == 2
