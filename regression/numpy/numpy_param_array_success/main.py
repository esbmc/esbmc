import numpy as np

def get_row(a):
    row = a[0]
    return row[0]

a = np.array([[1, 2], [3, 4]])
first = get_row(a)
assert first == 1
