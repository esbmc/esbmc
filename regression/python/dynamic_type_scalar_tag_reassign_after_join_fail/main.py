cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"
x = 2
assert x == 3
