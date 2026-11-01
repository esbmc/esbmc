import numpy as np

a = np.array([3, 1, 4, 1, 5])

assert a.argmin() == 1
assert a.argmax() == 4

a_alt = np.array([2, 1, 3, 1, 2])

assert a_alt.argmin() == 1
assert a_alt.argmax() == 2

f_m1 = np.array([1.5, 2.5, -1.5])
truncated_m1 = 0
for x_m1 in np.nditer(f_m1, op_dtypes=["int64"], flags=["buffered"], casting="unsafe"):
    truncated_m1 = truncated_m1 * 10 + x_m1
assert truncated_m1 == 119

b_m1 = np.array([100, -100], dtype=np.int8)
for e_m1 in np.nditer(b_m1, op_flags=["readwrite"]):
    e_m1[...] = e_m1 + 100
assert b_m1[0] == -56 and b_m1[1] == 0
