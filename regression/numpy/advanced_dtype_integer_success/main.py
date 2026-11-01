import numpy as np

a = np.array([7, -3], dtype="int16")
b = np.ones(3, dtype="uint32")
assert a[0] == 7 and a[1] == -3
assert b[2] == 1

c = np.array([127, -128], dtype=np.int8)
d = np.zeros(2, dtype=np.uint8)
e = np.array([0.1, 2.5], dtype=np.float32)
c[0] += 1
d[1] = d[1] - 1
assert c[0] == -128
assert d[1] == 255
assert float(e[0]) != 0.1
assert e[1] == 2.5

assert c.dtype == np.int8
assert c.itemsize == 1 and c.nbytes == 2
assert e.itemsize == 4 and e.nbytes == 8
assert b.dtype == "uint32"
assert b.itemsize == 4 and b.nbytes == 12

a_m1 = np.array([1, 2], dtype="float32")
b_m1 = np.zeros((2, 2), dtype=np.float16)
assert a_m1[1] == 2.0
assert b_m1[1][0] == 0.0

narrow_values_m1 = np.array([0.1, 0.2], dtype=np.float32)
assert narrow_values_m1.mean() == 0.15000000596046448
short_values_m1 = np.zeros(1, dtype=np.int16)
assert short_values_m1.dtype.name == "int16"

a_m2 = np.full(3, 2.5, dtype=np.int64)
c_m2 = a_m2.copy()
a_m2[0] = 9
assert c_m2[0] == 2 and a_m2[0] == 9

b_m2 = np.array([100, -100], dtype=np.int8)
d_m2 = b_m2.copy()
d_m2[0] += 100
assert d_m2[0] == -56 and b_m2[0] == 100 and d_m2.itemsize == 1

a_m3=np.array([3],dtype=np.int16)
assert int(a_m3[0])==3

a_m4=np.array([1],dtype=np.uint8)
assert a_m4.dtype==np.uint8
b_m4=np.array([1j],dtype=np.complex64)
assert b_m4.dtype==np.complex64
c_m4=np.array([True])
assert c_m4.dtype==bool

a_m5=np.array([[1,2]],dtype=np.int64)
b_m5=np.array([1],dtype=np.int32)
c_m5=a_m5[0][1]+b_m5[0]
assert c_m5==3

a_m6=np.array([1.0],dtype=np.float32)
b_m6=sum(a_m6)
assert b_m6==1.0
