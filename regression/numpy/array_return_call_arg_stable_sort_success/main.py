import numpy as np

call_count = 0


def bump():
    global call_count
    call_count = call_count + 1
    return call_count


def make(n):
    a = np.array([3, 1, 2])
    return a


made = make(bump())
result = np.sort(made, kind="stable")
assert result[0] == 1
assert call_count == 1
