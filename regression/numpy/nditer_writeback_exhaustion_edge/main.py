import numpy as np

a = np.array([1, 2, 3], dtype=np.int64)
it = np.nditer(a, flags=["buffered"], op_flags=[["readwrite"]],
               op_dtypes=["float64"], casting="unsafe")
it[0][...] = 9.9
assert a[0] == 1
it.close()
assert a[0] == 9

b = np.array([1, 2, 3], dtype=np.int64)
for x in np.nditer(b, flags=["buffered"], op_flags=[["readwrite"]],
                   op_dtypes=["float64"], casting="unsafe"):
    x[...] = 9.9
    break
assert b[0] == 1

c = np.array([1, 2], dtype=np.int64)
for x in np.nditer(c, flags=["buffered"], op_flags=[["readwrite"]],
                   op_dtypes=["float64"], casting="unsafe"):
    x[...] = 4.5
assert c[1] == 4
