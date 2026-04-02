z = complex(1, 2)
z += 3
assert z == complex(4, 2)

z -= 2
assert z == complex(2, 2)

z2 = complex(3, 6)
z2 *= 2
assert z2 == complex(6, 12)

z3 = complex(4, 2)
z3 /= 2.0
assert z3 == complex(2.0, 1.0)

z4 = complex(1, 2)
z4 += 1.5
assert z4 == complex(2.5, 2.0)

z5 = complex(0, 0)
z5 += 0
assert z5 == complex(0, 0)

z6 = complex(2, 3)
z6 += True
assert z6 == complex(3, 3)
