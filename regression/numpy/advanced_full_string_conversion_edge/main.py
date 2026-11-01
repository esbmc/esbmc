import numpy as np

a = np.full(2, 5, dtype=str)
assert a[0] == "5"
b = np.full(2, 123, dtype=str)
assert b[1] == "1" and b.itemsize == 4
c = np.full(2, 123, dtype="U2")
assert c[0] == "12"
d = np.array([12, 3], dtype=str)
assert d[0] == "12" and d[1] == "3"

literal_fill = np.full(2, "ab", dtype=str)
assert literal_fill[1] == "a"
assert np.full(2, "ab")[1] == "ab"
