import numpy as np

calls = 0


def first():
    global calls
    calls = calls * 10 + 1
    return np.array([1.0, 2.0])


def second():
    global calls
    calls = calls * 10 + 2
    return np.array([3.0, 4.0])


def mask():
    global calls
    calls = calls * 10 + 3
    return np.array([True, False])


o = np.array([9.0, 9.0])
np.add(first(), second(), out=o, where=mask())
assert calls == 123
assert o[0] == 4.0 and o[1] == 9.0
