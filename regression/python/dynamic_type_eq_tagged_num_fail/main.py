# The numeric arm of the tagged-vs-tagged compare, isolated: the pair is equal
# on the str path, so only a wrong numeric compare could make this hold.

cond = nondet_bool()
if cond:
    x = 1
    y = 2
else:
    x = "ab"
    y = "ab"
assert x == y
