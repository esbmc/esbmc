# The provably-false answer is asserted, not assumed away.

cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"
assert isinstance(x, list)
