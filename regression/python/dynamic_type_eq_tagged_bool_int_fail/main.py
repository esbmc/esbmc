cond = nondet_bool()
if cond:
    x = True
    y = 2
else:
    x = "a"
    y = "b"
if cond:
    assert x == y
