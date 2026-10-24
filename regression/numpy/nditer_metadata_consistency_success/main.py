import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([10, 20, 30])
it = np.nditer([a, b], flags=["multi_index", "c_index"])
assert it.ndim == 2
assert it.shape == (2, 3)
assert it.itersize == 6
assert it.nop == 2
while not it.finished:
    i, j = it.multi_index
    assert it.index == i * 3 + j
    assert it.iterindex == i * 3 + j
    assert it[0] == a[i][j]
    assert it[1] == b[j]
    it.iternext()

f = np.nditer(a, flags=["f_index"])
order = []
while not f.finished:
    order.append(f.index)
    f.iternext()
assert order == [0, 2, 4, 1, 3, 5]
