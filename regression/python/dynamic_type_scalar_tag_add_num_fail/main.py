cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"
    y = x + 1
    assert y == 2
