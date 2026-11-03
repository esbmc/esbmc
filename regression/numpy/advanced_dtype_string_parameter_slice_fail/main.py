import numpy as np


def f(x):
    b = x[0:2]
    x[0] = "zz"
    return b[0]


a = np.array(["ab", "cd", "ef"])
assert f(a) == "zz"
