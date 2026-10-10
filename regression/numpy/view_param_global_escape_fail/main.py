import numpy as np

_stored = None

def store_view(row):
    global _stored
    _stored = row

a = np.array([1, 2, 3])
v = a[0:]
store_view(v)
