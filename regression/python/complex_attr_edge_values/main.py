# Tests .real and .imag with edge values: inf, large, small

# Infinity in real part
z1 = complex(float("inf"), 1.0)
assert z1.real == float("inf")
assert z1.imag == 1.0

# Infinity in imag part
z2 = complex(2.0, float("inf"))
assert z2.real == 2.0
assert z2.imag == float("inf")

# Negative infinity
z3 = complex(float("-inf"), float("-inf"))
assert z3.real == float("-inf")
assert z3.imag == float("-inf")

# Very large values
z4 = complex(1e308, -1e308)
assert z4.real == 1e308
assert z4.imag == -1e308

# Very small values
z5 = complex(1e-308, -1e-308)
assert z5.real == 1e-308
assert z5.imag == -1e-308

# Mixed inf and zero
z7 = complex(float("inf"), 0.0)
assert z7.real == float("inf")
assert z7.imag == 0.0
