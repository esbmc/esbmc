import numpy as np

call_count = 0


def make():
    global call_count
    call_count += 1
    a = np.array([1, 3, 5, 7])
    return a


i = np.searchsorted(make(), 4)
assert i == 2
assert call_count == 1
