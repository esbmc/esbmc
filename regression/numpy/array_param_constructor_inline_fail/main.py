import numpy as np


def total(a):
    s = 0.0
    for v in a:
        s += v
    return s


assert total(np.arange(1.0, 4.0)) == 7.0
