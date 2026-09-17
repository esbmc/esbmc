# The two tagged operands hold different Python types on either path. Python
# compares across types as unequal rather than coercing.

cond = nondet_bool()
if cond:
    x = 1
    y = "a"
else:
    x = "a"
    y = 1
assert x != y
