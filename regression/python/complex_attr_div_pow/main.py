# Tests .real/.imag after division and pow operations

# After division
z1 = complex(4.0, 2.0) / complex(1.0, 1.0)
# (4+2j)/(1+1j) = (4+2j)(1-1j)/((1+1j)(1-1j)) = (4-4j+2j-2j^2)/(1+1) = (6-2j)/2 = 3-1j
assert z1.real == 3.0
assert z1.imag == -1.0

# After pow with int exponent
z2 = complex(0.0, 1.0)
z2_sq = z2 ** 2
# i^2 = -1
assert z2_sq.real == -1.0
assert z2_sq.imag == 0.0

# After pow: (1+1j)^2 = 2j
z3 = complex(1.0, 1.0) ** 2
assert z3.real == 0.0
assert z3.imag == 2.0

# Division by real gives scaled components
z4 = complex(10.0, -6.0) / complex(2.0, 0.0)
assert z4.real == 5.0
assert z4.imag == -3.0
