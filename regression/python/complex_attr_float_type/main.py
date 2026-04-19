# Tests .real and .imag return float type by verifying float operations

# .real participates in float arithmetic
z1 = complex(3.0, 4.0)
r = z1.real
half_r = r / 2.0
assert half_r == 1.5

# .imag participates in float arithmetic
i = z1.imag
half_i = i / 2.0
assert half_i == 2.0

# .real + .imag gives float
total = z1.real + z1.imag
assert total == 7.0

# .real * .imag gives float
product = z1.real * z1.imag
assert product == 12.0

# .real - .imag gives float
diff = z1.real - z1.imag
assert diff == -1.0

# Components used to build a new complex (round-trip)
z2 = complex(z1.real, z1.imag)
assert z2 == z1

# Components compared with float literals
assert z1.real == 3.0
assert z1.imag == 4.0
