import numpy as np


def value() -> int:
    return 9


a = np.array([[1, 2], [3, 4]])
t = np.transpose(a)

t[0][1] = value()

assert a[1][0] == 9
