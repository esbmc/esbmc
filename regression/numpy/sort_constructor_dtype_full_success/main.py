import numpy as np

a = np.sort(np.full((3,), 5.7, dtype=int))

assert a[0] == 5
assert a[1] == 5
assert a[2] == 5

b = np.sort(np.array([100, -100], dtype=np.int8))
assert b[0] == -100 and b.itemsize == 1
b[1] += 100
assert b[1] == -56

# Without the dtype=int truncation the array holds 5.7 and searching for the
# untruncated 5.7 would insert at index 0 (left of the matching elements);
# truncated to 5, 5.7 sorts strictly after every element instead.
i_m1 = np.searchsorted(np.full((3,), 5.7, dtype=int), 5.7)

assert i_m1 == 3

n_m1 = np.array([1, 2, 3], dtype=np.int8)
n_m1[2] += 127
assert np.searchsorted(np.array([1, 2, 3], dtype=np.int8), 2) == 1
assert n_m1[2] == -126

base_m2 = np.array([[1, 2], [3, 4]], dtype=int)
lst_m2 = base_m2.tolist()

assert lst_m2[0][0] == 1
assert lst_m2[0][1] == 2
assert lst_m2[1][0] == 3
assert lst_m2[1][1] == 4

narrow_m2 = np.array([100, -100], dtype=np.int8)
narrow_m2[0] += 100
assert narrow_m2.tolist()[0] == -56
single_m2 = np.array([0.1], dtype=np.float32)
assert single_m2.tolist()[0] != 0.1
