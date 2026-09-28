cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"
lst = [x]
assert lst[0] == 1 or lst[0] == "a"
