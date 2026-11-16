import numpy as np

z = np.zeros((2, 2), dtype="int8")
o = np.ones(3, dtype=np.uint16)
assert z[1][1] == 0
assert o.sum() == 3

e = np.eye(2, dtype="int32")
i = np.identity(3, dtype=np.bool_)
assert e[1][1] == 1 and e[0][1] == 0
assert i[2][2] and not i[0][2]

a = np.array([3.9], dtype=np.int_)
b = np.array([3.9], dtype="int64")
c = np.array([3.9], dtype=int)
d = np.array(["q"], dtype=np.str_)
assert a[0] == 3 and b[0] == 3 and c[0] == 3
assert d[0] == "q"

zi = np.zeros((2, 2), dtype=int)
assert zi[0][0] == 0 and zi[1][1] == 0
ob = np.ones((1, 2), dtype=bool)
assert ob[0][0] == True and ob[0][1] == True
fi = np.full((1,), 3.7, dtype=int)
assert fi[0] == 3

# A program's name shadows a dtype code of the same spelling.
f = np.int8
narrow = np.array([100], dtype=f)
twice = narrow + narrow
assert twice[0] == -56

# A value read from an int64 array is strong: the result is int64.
wide = np.array([28, 100])
strong = narrow + wide.min()
assert strong[0] == 128

a_m1 = np.linspace(0, 3, 4, dtype=int)
assert np.sum(a_m1) == 6
assert max(a_m1) == 3
assert a_m1.mean() == 1.5

b_m1 = np.array([100, 100], dtype=np.int8)
assert np.sum(b_m1) == 200
assert np.prod(b_m1) == 10000
assert np.mean(b_m1) == 100.0
assert sum(b_m1) == -56
c_m1 = np.add(b_m1, b_m1)
assert c_m1[0] == -56 and c_m1.itemsize == 1

a_alt = np.zeros((2, 1))
assert a_alt[0][0] == 0.0
assert a_alt[1][0] == 0.0

a_m1 = np.ones((2, 3))
assert a_m1[0][0] == 1.0
assert a_m1[1][2] == 1.0

p_m1 = np.ones((1, 2))
assert p_m1[0][0] == 1.0
assert p_m1[0][1] == 1.0

a_m2 = np.zeros((2, 3))
s_m2 = a_m2.shape
assert s_m2[0] == 2
assert s_m2[1] == 3

a_alt_m2 = np.zeros((1, 1))
s_alt_m2 = a_alt_m2.shape
assert s_alt_m2[0] == 1
assert s_alt_m2[1] == 1
