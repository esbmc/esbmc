import numpy as np

def select(flag):
    a = np.array([1, 2, 3])
    b = np.array([10, 20])
    if flag:
        v = a[0:]
    else:
        v = b[0:]
    return v[0]

result = select(True)
assert result == 1
