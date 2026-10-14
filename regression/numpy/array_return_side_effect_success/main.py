import numpy as np

calls = 0


def make():
    global calls
    calls = calls + 1
    a = np.array([10, 20, 30])
    return a


a1 = make()
assert calls == 1
assert a1[0] == 10

a2 = make()
assert calls == 2
assert a2[1] == 20
