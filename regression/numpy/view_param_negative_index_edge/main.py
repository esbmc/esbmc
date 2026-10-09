import numpy as np

def last(row):
    return row[-1]

a = np.array([10, 20, 30])
v = a[0:]
result = last(v)
assert result == 30
