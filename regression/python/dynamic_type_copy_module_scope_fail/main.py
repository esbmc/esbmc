cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"
y = x
assert y == 1
