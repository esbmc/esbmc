# The join of a float and a bool binding keeps the bool's value.
cond = nondet_bool()
if cond:
    y = 2.5
else:
    y = True
assert y == (2.5 if cond else 1)
