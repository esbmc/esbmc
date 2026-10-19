import numpy as np


def pick(wide: bool):
    kind = int if wide else float
    return np.zeros(2, dtype=kind)


a = pick(True)
assert a[0] == 0
