# Both branches assign the same type (int), so there is no cross-type join to lose.
# The post-join assertion is correctly checked on both paths.

cond = nondet_bool()
if cond:
    x = 1
    y = 2
else:
    x = 3
    y = 4
z = x + y
assert z == 3 or z == 7
