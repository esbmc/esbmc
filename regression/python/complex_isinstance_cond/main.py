# isinstance(z, complex) with result used in conditional.

z = complex(1, 2)
x = 42

# Complex variable passes isinstance check for complex.
if isinstance(z, complex):
    r = z.real
else:
    r = 0.0
assert abs(r - 1.0) < 1e-10

# Non-complex variable fails isinstance check for complex.
if isinstance(x, complex):
    r2 = 1.0
else:
    r2 = 0.0
assert r2 == 0.0
