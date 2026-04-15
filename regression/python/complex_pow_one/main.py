# z**1 should return z unchanged.
z = complex(2.5, -3.5)
w = z ** 1
assert abs(w.real - 2.5) < 1e-10
assert abs(w.imag - (-3.5)) < 1e-10
