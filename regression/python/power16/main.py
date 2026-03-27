# Perfect roots
assert 8**(1 / 3) == 2.0  # cube root
assert 27**(1 / 3) == 3.0  # cube root
assert 16**(0.25) == 2.0  # fourth root
assert 32**(1 / 5) == 2.0  # fifth root
assert 4**0.5 == 2.0  # square root

# Negative exponents (reciprocals)
assert 8**(-1 / 3) == 0.5  # 1/(cube root)

# Non-perfect cases
assert abs(10**(1 / 3) - 2.154434690031884) < 1e-10

# Powers of non-integers
assert abs(2.5**2.5 - 9.882117688026186) < 1e-10

# Edge cases that shouldn't trigger rounding
assert abs(7**(1 / 3) - 1.912931182772389) < 1e-10
