import numpy as np

def f(x):
    return x[0]


a = np.array([127], dtype=np.int8)
assert f(a) + 1 == -128
