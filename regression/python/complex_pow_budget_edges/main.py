# Edge cases for budget limit: boundary values.

# Exponent exactly at boundary (16) with non-trivial base.
z1 = complex(0, 1)
w1 = z1 ** 16
# i^16 = (i^4)^4 = 1^4 = 1
assert abs(w1.real - 1.0) < 1e-6
assert abs(w1.imag - 0.0) < 1e-6

# Exponent just beyond boundary (17) uses exp/log fallback.
z2 = complex(0, 1)
w2 = z2 ** 17
# i^17 = i^16 * i = i
assert abs(w2.real - 0.0) < 1e-4
assert abs(w2.imag - 1.0) < 1e-4
