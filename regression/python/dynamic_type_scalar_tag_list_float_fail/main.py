cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"
if cond:
    x = 1.5
lst = []
lst.append(x)
assert isinstance(lst[0], float)
