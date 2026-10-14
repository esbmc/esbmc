cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"
lst = [x]
lst.insert(0, x)
assert lst[0] == 1 or lst[0] == "a"
assert lst[1] == 1 or lst[1] == "a"
