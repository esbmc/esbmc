# Tests that ESBMC detects incorrect conjugate assertion
z = complex(3.0, 4.0)
c = z.conjugate()
assert c.imag == 4.0  # wrong: should be -4.0
