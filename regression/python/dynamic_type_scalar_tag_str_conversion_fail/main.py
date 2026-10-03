cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"

y = str(x)
assert y == "2"
