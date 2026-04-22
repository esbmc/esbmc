# Tests that ESBMC detects incorrect .imag assertion
z = complex(3.0, 4.0)
assert z.imag == 999.0
