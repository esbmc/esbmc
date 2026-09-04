import numpy as np

calls = 0


def make():
    global calls
    calls = calls + 1
    return np.array([3, 1, 2])


y1 = make()
s = y1.sum()

y2 = make()
mx = y2.max()

assert s == 6
assert mx == 3
assert calls == 2
