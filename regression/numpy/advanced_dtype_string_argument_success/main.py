import numpy as np


def first(x):
    return x[0]


def count(x):
    return len(x)


def overwrite(x):
    x[0] = "zzz"


a = np.array(["ab", "c"])
assert first(a) == "ab"
assert count(a) == 2
assert "c" in a
assert "d" not in a
assert list(a)[1] == "c"
assert len(tuple(a)) == 2
overwrite(a)
assert a[0] == "zz"
