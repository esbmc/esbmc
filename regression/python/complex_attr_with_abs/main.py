# Tests .real/.imag with abs() — verifies abs returns float and components work

# abs of Pythagorean triple
z1 = complex(3.0, 4.0)
mag1 = abs(z1)
assert mag1 == 5.0

# abs result used in arithmetic with .real
z2 = complex(3.0, 4.0)
ratio = abs(z2) / z2.real  # 5 / 3
r = ratio * 3.0
assert r == 5.0

# abs result compared directly with component magnitude
z3 = complex(0.0, 5.0)
assert abs(z3) == z3.imag  # |0+5j| = 5 == 5

# abs of purely real
z4 = complex(7.0, 0.0)
assert abs(z4) == z4.real

# abs after conjugate is same as abs before
z5 = complex(3.0, 4.0)
c5 = z5.conjugate()
assert abs(c5) == abs(z5)
