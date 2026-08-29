cond = nondet_bool()
if cond:
    x = 10
    y = x / 0
    assert y == 5
else:
    x = "a"
