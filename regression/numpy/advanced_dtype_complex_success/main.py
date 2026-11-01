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
