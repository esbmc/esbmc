import numpy as np

call_count = 0


def bump():
    global call_count
    call_count = call_count + 1
    return call_count


def make_array(n):
    a = np.array([5, 10, 15])
    return a


result = make_array(bump())
# Wrong assertion: side effect should run exactly once, not twice
assert call_count == 2
