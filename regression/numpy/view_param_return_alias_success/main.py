import numpy as np

def alias_and_return(row):
    alias = row
    alias[1] = 99
    return alias

a = np.array([1, 2, 3])
v = a[0:]
ret = alias_and_return(v)
assert a[1] == 99
