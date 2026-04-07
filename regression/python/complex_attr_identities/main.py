# Tests .real and .imag used to verify mathematical identities

# |z|^2 == z.real^2 + z.imag^2
z1 = complex(3.0, 4.0)
mag = abs(z1)
mag_sq = mag * mag
component_sq = z1.real * z1.real + z1.imag * z1.imag
assert mag_sq == component_sq

# z + conj(z) == 2 * z.real (purely real)
z2 = complex(5.0, 7.0)
s = z2 + z2.conjugate()
assert s.real == 2.0 * z2.real
assert s.imag == 0.0

# z - conj(z) == 2j * z.imag (purely imaginary)
z3 = complex(5.0, 7.0)
d = z3 - z3.conjugate()
assert d.real == 0.0
assert d.imag == 2.0 * z3.imag

# Conjugate of sum == sum of conjugates
a = complex(1.0, 2.0)
b = complex(3.0, 4.0)
conj_sum = (a + b).conjugate()
sum_conj = a.conjugate() + b.conjugate()
assert conj_sum == sum_conj
