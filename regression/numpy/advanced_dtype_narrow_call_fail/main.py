import numpy as np

def f(x):
    return x + 1


a = np.array([127], dtype=np.int8)
assert f(a[0]) == -128
