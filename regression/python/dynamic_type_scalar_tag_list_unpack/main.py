cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"
lst = [x, x]
a, *b = lst
assert a == 1 or a == "a"
assert b[0] == 1 or b[0] == "a"
