import numpy as np

def get_first(row):
    return row[0]

a = np.array([10, 20, 30])
b = np.array([40, 50, 60])
va = a[0:]
vb = b[0:]
ra = get_first(va)
rb = get_first(vb)
assert ra == 40
