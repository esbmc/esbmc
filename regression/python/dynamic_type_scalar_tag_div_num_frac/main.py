cond = nondet_bool()
if cond:
    x = 7
    y = x / 2
    assert y == 3.5
else:
    x = "a"
