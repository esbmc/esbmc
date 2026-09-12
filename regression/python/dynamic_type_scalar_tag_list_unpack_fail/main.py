cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"
lst = [x, x]
a, *b = lst
assert b[0] == 1
