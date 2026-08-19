# The str arm, isolated: the pair is equal on the numeric path, so only a
# wrong byte compare could make this hold.

cond = nondet_bool()
if cond:
    x = 1
    y = 1
else:
    x = "ab"
    y = "ac"
assert x == y
