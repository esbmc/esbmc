cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"
lst = [x]
other = lst + lst
assert other[1] == 1
