# The float check is evaluated, not folded: asserting the wrong direction is
# detected.

cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"
x = 1.5
assert not isinstance(x, float)
