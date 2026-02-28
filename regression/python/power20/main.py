# Perfect roots
assert 8**(1 / 3) == 2.0
assert 27**(1 / 3) == 3.0
assert 16**(1 / 4) == 2.0
assert 32**(1 / 5) == 2.0

# Imperfect roots (within tolerance)
assert abs(10**(1 / 3) - 2.154434690031884) < 1e-10
assert abs(7**(1 / 3) - 1.912931182772389) < 1e-10

# Reciprocal roots
assert 8**(-1 / 3) == 0.5
