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
