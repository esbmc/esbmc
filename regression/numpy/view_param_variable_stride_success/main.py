import numpy as np

def read_two(row):
    x = row[0]
    y = row[1]
    return x + y

a = np.array([1, 2, 3, 4, 5, 6])
step = 2
v = a[::step]
result = read_two(v)
assert result == 4
