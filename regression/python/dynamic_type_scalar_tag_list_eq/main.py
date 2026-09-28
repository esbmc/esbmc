cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"
a = [x]
b = [x]
assert a == b
