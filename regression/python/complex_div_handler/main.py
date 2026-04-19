# Tests complex division in isolation with known exact results
# Exercises the handle_binary_op Div -> complex_div formula

# (3+4j) / (1+2j) = (3+4j)(1-2j) / (1+4) = (3+8+4j-6j)/5 = (11-2j)/5 = 2.2-0.4j
z = complex(3, 4) / complex(1, 2)
assert z.real > 2.19 and z.real < 2.21
assert z.imag > -0.41 and z.imag < -0.39

# Division by purely imaginary: (3+4j) / (0+2j) = (3+4j)(-2j) / (4) = (-6j+8)/4 = 2-1.5j
w = complex(3, 4) / complex(0, 2)
assert w.real > 1.99 and w.real < 2.01
assert w.imag > -1.51 and w.imag < -1.49

# Division by purely real: (3+4j) / (2+0j) = 1.5 + 2j
v = complex(3, 4) / complex(2, 0)
assert v.real > 1.49 and v.real < 1.51
assert v.imag > 1.99 and v.imag < 2.01

# Identity division: z / (1+0j) == z
z2 = complex(5, 7)
r = z2 / complex(1, 0)
assert r.real > 4.99 and r.real < 5.01
assert r.imag > 6.99 and r.imag < 7.01
