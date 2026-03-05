z0 = complex()
assert z0.real == 0.0
assert z0.imag == 0.0

z1 = complex(1)
assert z1.real == 1.0
assert z1.imag == 0.0

z2 = complex(1, 2)
assert z2.real == 1.0
assert z2.imag == 2.0

z3 = complex(True, False)
assert z3.real == 1.0
assert z3.imag == 0.0
