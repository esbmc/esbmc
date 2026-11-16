import numpy as np

a = np.array([1 + 2j, 3j])
b = np.ones(2, dtype=complex)
assert a[0] == 1 + 2j
assert a[1].imag == 3.0
assert b[1] == 1

zero = np.zeros(2, dtype=np.complex64)
mixed = np.array([0j, 2])
assert zero[0] == 0j
assert zero[1].imag == 0.0
assert mixed[1] == 2 and mixed[1].imag == 0.0

filled = np.full(2, 1 - 1j)
fill_real = np.full(2, 3, dtype=complex)
assert filled[1] == 1 - 1j
assert fill_real[0] == 3 and fill_real[0].imag == 0.0

# complex64 computes and compares at its component width.
big = np.array([3e38 + 3e38j], dtype=np.complex64)
scale = np.array([1e38 + 1e38j], dtype=np.complex64)
ratio = big / scale
assert ratio[0].real == float("inf")
tenth = np.array([0.1 + 0.1j], dtype=np.complex64)
assert tenth[0] == 0.1 + 0.1j
pure = np.array([0.1 + 0j], dtype=np.complex64)
shifted = pure[0] + 0.1
assert shifted.real == 0.2
quotient = np.array([1 + 2j], dtype=np.complex64) / np.array([3 + 4j], dtype=np.complex64)
assert quotient[0].real == 0.44
assert quotient[0].imag == 0.08
single = np.array([0.5 + 1j, 1j], dtype=np.complex64)
assert single.itemsize == 8
assert single.astype(complex)[1] == 1j
assert np.array([1, 2]).astype(np.complex64)[1] == 2
held = list(single)
assert held[1] == 1j
assert single.tolist()[0] == 0.5 + 1j

a_m1 = np.array([1 + 2j, 3j])
b_m1 = np.array([1j, 2])
c_m1 = a_m1 * 2
assert c_m1[0] == 2 + 4j and c_m1[1] == 6j
d_m1 = a_m1 + b_m1
assert d_m1[0] == 1 + 3j and d_m1[1] == 2 + 3j
e_m1 = a_m1 - b_m1
assert e_m1[1] == -2 + 3j
f_m1 = a_m1 * b_m1
assert f_m1[0] == -2 + 1j
g_m1 = a_m1 / b_m1
assert g_m1[0] == 2 - 1j and g_m1[1] == 1.5j
h_m1 = -a_m1
assert h_m1[0] == -1 - 2j
i_m1 = a_m1 + 1.5
assert i_m1[1] == 1.5 + 3j
j_m1 = 1j * a_m1
assert j_m1[1] == -3 + 0j
k_m1 = np.add(a_m1, b_m1)
assert k_m1[1] == 2 + 3j
l_m1 = np.multiply(a_m1, 2)
assert l_m1[0] == 2 + 4j
n_m1 = np.array([1 + 1j, 1j]) / np.array([0, 1])
assert n_m1[1] == 1j
m_m1 = np.array([[1j, 2], [3, 4j]])
o_m1 = m_m1 + np.array([1, 1j])
assert o_m1[1][1] == 5j and o_m1.nbytes == 64

s_m1 = np.array([0.1 + 0.2j], dtype=np.complex64)
assert s_m1.itemsize == 8 and s_m1.nbytes == 8 and s_m1.dtype == np.complex64
t_m1 = s_m1 * 2
assert t_m1.itemsize == 8 and t_m1.dtype == "c8"
u_m1 = s_m1 + np.array([1j])
assert u_m1.itemsize == 16
assert float(s_m1[0].real) != 0.1
widened_m1 = np.array([1, 2, 3]).astype(np.complex128)
assert widened_m1[1] == 2 + 0j
assert widened_m1.dtype == np.complex128 and widened_m1.itemsize == 16 and widened_m1.nbytes == 48
from_text_m1 = np.array([1.5]).astype("c16")
assert from_text_m1[0].real == 1.5 and from_text_m1[0].imag == 0.0

def first_m2(x):
    return x[0]


def doubled_m2(x):
    return x * 2


a_m2 = np.array([1 + 2j, 3j])
assert np.sum(a_m2) == 1 + 5j
assert a_m2.sum() == 1 + 5j
assert sum(a_m2) == 1 + 5j
assert np.prod(a_m2) == -6 + 3j
assert a_m2.prod() == -6 + 3j
assert np.mean(a_m2) == 0.5 + 2.5j
assert np.any(a_m2) and np.all(a_m2)
assert not np.all(np.array([0j, 1j]))
assert not np.any(np.array([0j, 0j]))
assert np.sum(np.array([1 + 2j, 3j])) == 1 + 5j

assert first_m2(a_m2) == 1 + 2j
d_m2 = doubled_m2(a_m2)
assert d_m2[1] == 6j

m_m2 = np.array([[1j, 2], [3, 4j]])
assert np.sum(m_m2) == 5 + 5j
assert np.sum(m_m2[0]) == 2 + 1j
v_m2 = a_m2[1:]
assert np.sum(v_m2) == 3j
v_m2[0] = 5j
assert a_m2[1] == 5j
c_m2 = a_m2.copy()
c_m2[0] = 9
assert a_m2[0] == 1 + 2j and c_m2[0] == 9
r_m2 = m_m2.reshape(4)
assert r_m2[3] == 4j
t_m2 = m_m2.T
assert t_m2[0][1] == 3
s_m2 = np.array([1 + 1j, 1j], dtype=np.complex64)
assert np.sum(s_m2) == 1 + 2j


def second_m2(x):
    return x[1]


imaginary_m2 = np.array([1j, 2j])
assert second_m2(imaginary_m2) == 2j
held_m2 = second_m2(imaginary_m2)
assert held_m2 == 2j

values_m2 = np.array([1 + 1j, 2j])
part_m2 = values_m2[0:2]
assert part_m2.sum() == 1 + 3j
assert values_m2[0:2].prod() == -2 + 2j
assert part_m2.mean() == 0.5 + 1.5j
assert (part_m2 + part_m2)[1] == 4j
same_m2 = +values_m2
same_m2[0] = 0j
assert same_m2[0] == 0j and values_m2[0] == 1 + 1j
endless_m2 = np.array([complex(float("inf"), 0)])
assert (-endless_m2)[0].real == -float("inf")

x_m3=np.array([1],dtype=np.int32)*1j
assert x_m3[0]==1j
y_m3=np.array([1],dtype=np.float32)*1j
assert y_m3[0]==1j
z_m3=np.array([1j],dtype=np.complex64)*1.5
assert z_m3[0]==1.5j

def f_m4(a):
    return a[0]
assert f_m4(np.array([1,2],dtype=np.complex128))==1
