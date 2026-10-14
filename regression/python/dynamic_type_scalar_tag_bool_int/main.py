cond = nondet_bool()
if cond:
    x = True
else:
    x = "a"
if cond:
    assert x == 1
