# Tests .real and .imag used in comparisons and conditions

# Compare components directly
z1 = complex(5.0, 3.0)
assert z1.real > z1.imag

# Component equality
z2 = complex(4.0, 4.0)
assert z2.real == z2.imag

# Component in boolean context
z3 = complex(1.0, 0.0)
assert z3.real != 0.0

# Sum of squared components equals abs squared
z4 = complex(3.0, 4.0)
mag_sq = z4.real * z4.real + z4.imag * z4.imag
assert mag_sq == 25.0

# Product of component and scalar
z5 = complex(2.0, 5.0)
doubled_real = z5.real * 2.0
assert doubled_real == 4.0

# Component difference
z6 = complex(10.0, 3.0)
diff = z6.real - z6.imag
assert diff == 7.0
