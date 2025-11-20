# Reversed operands: str < float
f = float(65)
assert not ('A' < f)

# String greater-than float
assert not ('B' > f)

# String less-than-equals float
assert not ('A' <= f)
