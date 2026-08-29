cond = nondet_bool()
if cond:
    x = 1
    y = x + 1
    assert y == 2
else:
    x = "a"
