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

c1 = complex(1, 2)
z4 = complex(c1)
assert z4.real == 1.0
assert z4.imag == 2.0

z5 = complex(c1, 3)
assert z5.real == 1.0
assert z5.imag == 5.0

c2 = complex(2, 3)
z6 = complex(1, c2)
assert z6.real == -2.0
assert z6.imag == 2.0

z7 = complex(c1, c2)
assert z7.real == -2.0
assert z7.imag == 4.0

raised = False
try:
    complex("1")
except TypeError:
    raised = True
assert raised

raised = False
try:
    complex(1, "2")
except TypeError:
    raised = True
assert raised
