cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"
    assert x < 5
