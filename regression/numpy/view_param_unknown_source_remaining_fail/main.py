import numpy as np

def set_first(row):
    row[0] = 9
    return len(row)

r = np.full(4, 7)
w = r[1:]
m = set_first(w)
assert r[1] == 9
