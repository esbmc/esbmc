# Tests .real/.imag across reassignment and sequential mutation

# Reassign complex, check components of each version
z = complex(1.0, 2.0)
r1 = z.real
i1 = z.imag
assert r1 == 1.0
assert i1 == 2.0

z = complex(10.0, 20.0)
r2 = z.real
i2 = z.imag
assert r2 == 10.0
assert i2 == 20.0

# Old captured values are unchanged
assert r1 == 1.0
assert i1 == 2.0

# Build a sequence of complex values and check components
a = complex(1.0, 0.0)
b = a + complex(0.0, 1.0)
c = b + complex(1.0, 0.0)
assert a.real == 1.0
assert a.imag == 0.0
assert b.real == 1.0
assert b.imag == 1.0
assert c.real == 2.0
assert c.imag == 1.0

# Component extracted before and after augmented op
z2 = complex(5.0, 3.0)
before_r = z2.real
z2 += complex(1.0, 1.0)
after_r = z2.real
assert before_r == 5.0
assert after_r == 6.0
