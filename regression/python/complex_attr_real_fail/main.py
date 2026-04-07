# Tests that ESBMC detects incorrect .real assertion
z = complex(3.0, 4.0)
assert z.real == 999.0
