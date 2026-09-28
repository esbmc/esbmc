cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"
lst = [x]
other = lst[:]
assert other[0] == 1
