import numpy as np

call_count = 0


def make_array():
    global call_count
    call_count = call_count + 1
    a = np.array([5, 10, 15])
    return a


def process(arr):
    return arr[0] + arr[1]


result = process(make_array())
assert result == 15
assert call_count == 1
