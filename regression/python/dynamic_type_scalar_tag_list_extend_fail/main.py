cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"
lst = [x]
lst.extend([x])
assert len(lst) == 1
