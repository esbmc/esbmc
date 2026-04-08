# z**0 should always be 1+0j for any non-zero z.
z = complex(3, 4)
w = z ** 0
assert w.real == 1.0
assert w.imag == 0.0
