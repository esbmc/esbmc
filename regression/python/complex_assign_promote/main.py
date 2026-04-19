z = complex(1, 2)
z = 3.14
assert z == complex(3.14, 0.0)

# Integer promotion
w = complex(5, 7)
w = 10
assert w == complex(10.0, 0.0)

# Zero float assignment
z2 = complex(4, 5)
z2 = 0.0
assert z2 == complex(0.0, 0.0)
