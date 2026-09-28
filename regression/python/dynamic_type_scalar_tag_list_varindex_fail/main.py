cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"
lst = [x]
i = 0
e = lst[i]
assert e == 1
