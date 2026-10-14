import numpy as np

def set_first(row):
    row[0] = 9
    return len(row)

r = np.array([1, 2, 3, 4], dtype=float)
w = r[1:3]
m = set_first(w)
