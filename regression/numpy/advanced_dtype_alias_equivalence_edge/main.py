import numpy as np

a = np.array([-2.7], dtype=int)
b = np.array([-2.7], dtype=np.int64)
c = np.array([-2.7], dtype="int64")
d = np.array([-2.7], dtype=np.int_)
assert a[0] == -2
assert a[0] == b[0] and b[0] == c[0] and c[0] == d[0]

m = np.array([0.5], dtype=np.bool_)
assert m[0] != 0.5 and m[0] == 1

e = np.array([1, 2], dtype="i1")
f = np.zeros(2, dtype="u2")
g = np.ones(2, dtype="f4")
h = np.array([1], dtype="<i4")
i = np.array([1], dtype="=i2")
j = np.array([255], dtype="B")
k = np.array([1.5], dtype="e")
assert e.dtype == np.int8 and e.itemsize == 1
assert f.dtype == "uint16" and f.itemsize == 2
assert g.dtype == np.float32 and g.nbytes == 8
assert h.dtype == "i4" and h.itemsize == 4
assert i.dtype == np.int16
assert j.dtype == np.uint8 and j[0] == 255
assert k.dtype == np.float16 and k.itemsize == 2

s = np.array(["abcd", "e"], dtype="U2")
assert s[0] == "ab" and s[1] == "e"
assert s.itemsize == 8 and s.nbytes == 16

narrow = np.int16
n = np.zeros(2, dtype=narrow)
assert n.itemsize == 2
code = "f4"
o = np.ones(1, dtype=code)
assert o.itemsize == 4
builtin = int
p = np.zeros(1, dtype=builtin)
assert p.itemsize == 8
