cond = nondet_bool()
if cond:
    x = 5
else:
    x = "a"
if not cond:
    x += 1
