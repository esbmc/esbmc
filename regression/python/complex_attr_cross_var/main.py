# Tests .real/.imag across multiple complex variables and cross-usage

# Access components from two different complex numbers
z1 = complex(1.0, 2.0)
z2 = complex(3.0, 4.0)
s = z1.real + z2.imag
assert s == 5.0

# Build new complex from components of two others
z3 = complex(z1.real, z2.imag)
assert z3.real == 1.0
assert z3.imag == 4.0

# Swap real and imag of a complex
z4 = complex(5.0, 7.0)
z4_swapped = complex(z4.imag, z4.real)
assert z4_swapped.real == 7.0
assert z4_swapped.imag == 5.0

# Compare components across variables
z5 = complex(10.0, 3.0)
z6 = complex(3.0, 10.0)
assert z5.real == z6.imag
assert z5.imag == z6.real

# Component arithmetic across variables
z7 = complex(2.0, 3.0)
z8 = complex(4.0, 5.0)
cross_product = z7.real * z8.imag - z7.imag * z8.real
assert cross_product == -2.0  # 2*5 - 3*4 = 10 - 12 = -2
