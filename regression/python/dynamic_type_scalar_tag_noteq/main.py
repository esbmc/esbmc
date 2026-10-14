cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"
if not cond:
    assert x != 1
