import numpy as np

def get_row(a):
    row = a[0]
    return row

a = np.array([[1, 2], [3, 4]])
row = get_row(a)
assert row[0] == 1
assert row[1] == 2
