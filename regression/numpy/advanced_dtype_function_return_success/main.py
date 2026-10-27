import numpy as np


def make():
    return np.arange(1.5, 4, dtype=np.int16)


a = make()
assert len(a) == 3
assert a[0] == 1 and a[2] == 3
