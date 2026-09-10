cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"
lst = []
lst.append(x)
assert lst[0] == 1 and lst[0] == "a"
