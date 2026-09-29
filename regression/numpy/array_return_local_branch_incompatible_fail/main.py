import numpy as np


def make(flag):
    if flag:
        a = np.zeros(3)
    else:
        a = np.zeros((2, 2))
    return a


y = make(True)

assert y[0] == 0
