import numpy as np


def first(x):
    return x[0]


def doubled(x):
    return x * 2


a = np.array([1 + 2j, 3j])
assert np.sum(a) == 1 + 5j
assert a.sum() == 1 + 5j
assert sum(a) == 1 + 5j
assert np.prod(a) == -6 + 3j
assert a.prod() == -6 + 3j
assert np.mean(a) == 0.5 + 2.5j
assert np.any(a) and np.all(a)
assert not np.all(np.array([0j, 1j]))
assert not np.any(np.array([0j, 0j]))
assert np.sum(np.array([1 + 2j, 3j])) == 1 + 5j

assert first(a) == 1 + 2j
d = doubled(a)
assert d[1] == 6j

m = np.array([[1j, 2], [3, 4j]])
assert np.sum(m) == 5 + 5j
assert np.sum(m[0]) == 2 + 1j
v = a[1:]
assert np.sum(v) == 3j
v[0] = 5j
assert a[1] == 5j
c = a.copy()
c[0] = 9
assert a[0] == 1 + 2j and c[0] == 9
r = m.reshape(4)
assert r[3] == 4j
t = m.T
assert t[0][1] == 3
s = np.array([1 + 1j, 1j], dtype=np.complex64)
assert np.sum(s) == 1 + 2j


def second(x):
    return x[1]


imaginary = np.array([1j, 2j])
assert second(imaginary) == 2j
held = second(imaginary)
assert held == 2j

values = np.array([1 + 1j, 2j])
part = values[0:2]
assert part.sum() == 1 + 3j
assert values[0:2].prod() == -2 + 2j
assert part.mean() == 0.5 + 1.5j
assert (part + part)[1] == 4j
same = +values
same[0] = 0j
assert same[0] == 0j and values[0] == 1 + 1j
endless = np.array([complex(float("inf"), 0)])
assert (-endless)[0].real == -float("inf")
