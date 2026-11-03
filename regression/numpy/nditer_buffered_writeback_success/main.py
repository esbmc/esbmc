import numpy as np

a = np.array([1, 2, 3], dtype=np.int64)
seen = []
with np.nditer(a, flags=["buffered"], op_flags=[["readwrite"]],
               op_dtypes=["float64"], casting="unsafe") as it:
    for x in it:
        seen.append(int(a[0]))
        x[...] = x * 2.6
    assert a[2] == 7
assert a[0] == 2
assert a[1] == 5
assert seen[2] == 1

b = np.array([1.0, 2.0, 3.0])
with np.nditer(b, flags=["buffered"], op_flags=[["readwrite"]]) as it:
    for x in it:
        x[...] = x * 10
        assert b[0] == 10.0
assert b[2] == 30.0
