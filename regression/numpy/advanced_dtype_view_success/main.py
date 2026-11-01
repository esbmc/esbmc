import numpy as np

a = np.array([1.9, 2.9, 3.9], dtype=np.int32)
v = a[1:]
assert v[0] == 2
a[2] = 5
assert v[1] == 5

b = np.array([100, -100, 50, 7], dtype=np.int8)
s = b[1:3]
s[0] = s[0] + s[0]
assert b[1] == 56 and s.itemsize == 1 and s.nbytes == 2
r = b[::-1]
r[0] += 121
assert b[3] == -128
m = np.array([[100, 27], [1, 2]], dtype=np.int8)
f = m.reshape(4)
f[0] += 100
assert m[0][0] == -56
t = m.T
t[0][1] = t[0][1] + 127
assert m[1][0] == -128
flat = b.ravel()
flat[0] += 100
assert b[0] == -56 and flat.itemsize == 1
first = b[1]
twice = first + first
assert twice == 112
minus = -m
assert minus[0][1] == -27 and minus.itemsize == 1

base = np.array([127, 5], dtype=np.int8)
window = base[0:1]
sum_view = window + 1
assert sum_view[0] == -128 and sum_view.itemsize == 1
window += 1
assert base[0] == -128
plain = np.array([1, 5])
plain_first = plain[0:1]
assert (plain_first + 1)[0] == 2 and (plain_first + plain_first)[0] == 2
grid = np.array([[1, 2], [3, 4]])
flipped = grid.T
bumped = flipped + 1
assert bumped[0][1] == 4
flipped += 1
assert grid[0][1] == 3
assert (base + base).itemsize == 1

a_m1 = np.array([-2.7], dtype=int)
b_m1 = np.array([-2.7], dtype=np.int64)
c_m1 = np.array([-2.7], dtype="int64")
d_m1 = np.array([-2.7], dtype=np.int_)
assert a_m1[0] == -2
assert a_m1[0] == b_m1[0] and b_m1[0] == c_m1[0] and c_m1[0] == d_m1[0]

m_m1 = np.array([0.5], dtype=np.bool_)
assert m_m1[0] != 0.5 and m_m1[0] == 1

e_m1 = np.array([1, 2], dtype="i1")
f_m1 = np.zeros(2, dtype="u2")
g_m1 = np.ones(2, dtype="f4")
h_m1 = np.array([1], dtype="<i4")
i_m1 = np.array([1], dtype="=i2")
j_m1 = np.array([255], dtype="B")
k_m1 = np.array([1.5], dtype="e")
assert e_m1.dtype == np.int8 and e_m1.itemsize == 1
assert f_m1.dtype == "uint16" and f_m1.itemsize == 2
assert g_m1.dtype == np.float32 and g_m1.nbytes == 8
assert h_m1.dtype == "i4" and h_m1.itemsize == 4
assert i_m1.dtype == np.int16
assert j_m1.dtype == np.uint8 and j_m1[0] == 255
assert k_m1.dtype == np.float16 and k_m1.itemsize == 2

s_m1 = np.array(["abcd", "e"], dtype="U2")
assert s_m1[0] == "ab" and s_m1[1] == "e"
assert s_m1.itemsize == 8 and s_m1.nbytes == 16

narrow_m1 = np.int16
n_m1 = np.zeros(2, dtype=narrow_m1)
assert n_m1.itemsize == 2
code_m1 = "f4"
o_m1 = np.ones(1, dtype=code_m1)
assert o_m1.itemsize == 4
builtin_m1 = int
p_m1 = np.zeros(1, dtype=builtin_m1)
assert p_m1.itemsize == 8
