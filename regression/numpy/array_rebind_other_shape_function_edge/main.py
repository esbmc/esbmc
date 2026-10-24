import numpy as np


def f() -> int:
    a = np.array([[1, 2], [3, 4]])
    total = int(a[1][1])
    a = np.array([5, 6, 7])
    return total + int(a[2]) + len(a)


assert f() == 14
