import numpy as np

def write_first(v):
    v[0] = 99

def make_sub_and_call(row):
    sub = row[1:]
    write_first(sub)
    return row[1]

a = np.array([1, 2, 3])
v = a[0:]
result = make_sub_and_call(v)
assert result == 99
