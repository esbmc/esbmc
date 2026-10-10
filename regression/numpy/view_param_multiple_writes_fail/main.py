import numpy as np

def write_both(row):
    row[0] = 11
    row[1] = 22

a = np.array([1, 2, 3])
v = a[0:]
write_both(v)
assert a[0] == 1  # wrong: write_both set a[0] to 11
