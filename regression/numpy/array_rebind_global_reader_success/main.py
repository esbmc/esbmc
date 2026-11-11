import numpy as np

a = np.array([1, 2])


def first() -> int:
    return int(a[0])


def size() -> int:
    return len(a)


assert first() == 1
c = nondet_bool()
if c:
    a = np.array([5, 6, 7])
assert first() == (5 if c else 1)
assert size() == (3 if c else 2)
for i in range(2):
    a = np.array([8, 9, 10, 11])
    assert first() == 8
assert size() == 4
