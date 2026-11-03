import numpy as np

m = np.array([[1, 2, 3], [4, 5, 6]])
order = 0
for x in np.nditer(m, op_axes=[[1, 0]], order="C"):
    order = order * 10 + x
assert order == 142536

it = np.nditer(m, op_axes=[[1, 0]], flags=["multi_index"])
for x in it:
    i, j = it.multi_index
    assert x == m[j, i]
