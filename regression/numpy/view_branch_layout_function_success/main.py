import numpy as np
def pick(cond):
    a = np.array([1, 2, 3, 4, 5])
    if cond:
        b = a[0:3]
    else:
        b = a[::2]
    b[1] = 0
    return len(b) + a[2]
assert pick(False) == 3
assert pick(True) == 6
