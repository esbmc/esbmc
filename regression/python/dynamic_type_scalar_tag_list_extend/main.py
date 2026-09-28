cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"
lst = [x]
lst.extend([x])
assert len(lst) == 2
assert lst[1] == 1 or lst[1] == "a"
