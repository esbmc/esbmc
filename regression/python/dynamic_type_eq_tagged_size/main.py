# Two tagged strings of different lengths, the longer on the left and holding
# an embedded NUL. Without the length check the byte compare would run past
# the end of the shorter payload instead of stopping.

cond = nondet_bool()
if cond:
    x = "a\x00b"
    y = "a"
else:
    x = 1
    y = 2
assert x != y
