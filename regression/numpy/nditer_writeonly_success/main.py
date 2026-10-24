import numpy as np

out = np.zeros(3, dtype=int)
n = 5
for x in np.nditer(out, op_flags=["writeonly"]):
    x[...] = n
    n += 1
assert out[0] == 5 and out[1] == 6 and out[2] == 7

src = np.array([1, 2])
dst = np.zeros(2, dtype=int)
for x, y in np.nditer([src, dst], op_flags=[["readonly"], ["writeonly"]]):
    y[...] = x * 10
assert dst[0] == 10 and dst[1] == 20
