import numpy as np

a = np.array([1, 2])
s1 = 0.0
for x in np.nditer(a, op_dtypes=[np.float64], flags=["buffered"]):
    s1 += x / 2
s2 = 0.0
for x in np.nditer(a, op_dtypes=["f8"], flags=["buffered"]):
    s2 += x / 2
assert s1 == 1.5 and s2 == 1.5

caught = False
try:
    np.nditer(np.array([1.5]), op_dtypes=["int64"], flags=["buffered"])
except TypeError:
    caught = True
assert caught
