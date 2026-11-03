import numpy as np

def set_first(row):
    row[0] = 9
    return len(row)

r = np.arange(5)
w = r[2:4]
m = set_first(w)
assert r[2] == 9
