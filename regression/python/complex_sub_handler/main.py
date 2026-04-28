# Tests complex subtraction in isolation
# Exercises the handle_binary_op Sub path

# Basic: (5+7j) - (2+3j) = (3+4j)
z = complex(5, 7) - complex(2, 3)
assert z == complex(3, 4)

# Producing negative components: (1+1j) - (3+5j) = (-2-4j)
w = complex(1, 1) - complex(3, 5)
assert w == complex(-2, -4)

# Subtracting zero: z - 0j = z
v = complex(3, 4) - complex(0, 0)
assert v == complex(3, 4)

# Mixed type: complex - int
r1 = complex(5, 3) - 2
assert r1 == complex(3, 3)

# Mixed type: complex - float
r3 = complex(5, 3) - 1.5
assert r3.real > 3.49 and r3.real < 3.51
assert r3.imag > 2.99 and r3.imag < 3.01
