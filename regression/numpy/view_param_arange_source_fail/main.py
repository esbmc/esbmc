import numpy as np
def set_first(row):
    row[0] = 9
    return len(row)
r = np.arange(1, 7, 2)
w = r[1:]
m = set_first(w)
assert m == 3
