cond = nondet_bool()
if cond:
    x = 5
else:
    x = "a"
if cond:
    x += 1
    assert x == 6
