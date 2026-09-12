cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"
lst = [x]
lst.insert(0, x)
assert lst[0] == 1
