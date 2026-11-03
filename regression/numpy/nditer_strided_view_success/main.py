import numpy as np

a = np.arange(8)
s = a[1::3]
it = np.nditer(s, flags=["c_index"], op_flags=[["readwrite"]])
indices = []
for x in it:
    indices.append(it.index)
    x[...] = -x
assert indices == [0, 1, 2]
assert a[1] == -1
assert a[4] == -4
assert a[7] == -7
assert a[2] == 2
