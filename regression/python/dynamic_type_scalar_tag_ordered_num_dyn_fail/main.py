cond = nondet_bool()
if cond:
    x = 5
    y = 5
    assert x < y
else:
    x = "a"
    y = "a"
