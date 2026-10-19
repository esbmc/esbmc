import numpy as np


def first(x):
    return x[0]


a = np.array(["ab", "c"])
assert first(a) == "ab"
