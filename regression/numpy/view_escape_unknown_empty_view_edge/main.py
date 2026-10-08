import numpy as np

calls = 0


def register(v):
    global calls
    calls = calls + 1


a = np.array([10, 20, 30, 40, 50])
v = a[2:2]
register(v)
assert len(v) == 0
assert v.shape == (0,)
assert v.size == 0
