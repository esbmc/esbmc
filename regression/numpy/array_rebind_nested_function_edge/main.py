import numpy as np


def keep() -> int:
    a = np.array([1, 2, 3])
    x = a[0:2]
    b = a
    a = np.array([0, 0, 0])
    x[1] = 7
    return int(b[1]) * 10 + int(a[1])


assert keep() == 70
