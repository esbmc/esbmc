import numpy as np

v = np.abs(3 + 4j)
assert v == 5.0
w = np.abs(0 - 7j)
assert w == 7.0

z_m1 = np.conjugate(5 + 6j)
assert z_m1.real == 5.0
assert z_m1.imag == -6.0
v_m1 = np.conj(10.0)
assert v_m1 == 10.0
ang_m1 = np.angle(0 + 0j)
assert ang_m1 == 0.0
assert np.real(7 - 9j) == 7.0
assert np.imag(7 - 9j) == -9.0
