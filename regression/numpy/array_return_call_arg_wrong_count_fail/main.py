import numpy as np

call_count = 0

def make_array():
    global call_count
    call_count = call_count + 1
    return np.array([5, 10, 15])


def process(arr):
    return arr[0] + arr[1]


result = process(make_array())
# Wrong assertion: if called twice, call_count would be 2
assert call_count == 2
