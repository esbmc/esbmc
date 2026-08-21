# The rebound list is read for real rather than assumed: a wrong element value
# is detected.

cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"
x = [1, 2, 3]
assert x[1] == 5
