import numpy as np

def first(a):
    return a[0]

arr = np.array([1, 2, 3])
assert first(arr) == 1

x = 5
y = first(x)
