# Comparing two tagged variables directly. Both branches give the pair equal
# values, so `x == y` holds on either path. Issue #7075.

cond = nondet_bool()
if cond:
    x = 1
    y = 1
else:
    x = "ab"
    y = "ab"
assert x == y
