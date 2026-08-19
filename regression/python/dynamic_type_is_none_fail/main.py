# The identity check is genuinely evaluated, not folded away: asserting the
# wrong direction is detected.

cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"
assert x is None
