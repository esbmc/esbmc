import numpy as np

z = np.conjugate(5 + 6j)
assert z.real == 5.0
assert z.imag == -6.0
v = np.conj(10.0)
assert v == 10.0
ang = np.angle(0 + 0j)
assert ang == 0.0
assert np.real(7 - 9j) == 7.0
assert np.imag(7 - 9j) == -9.0
