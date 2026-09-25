cond = nondet_bool()
if cond:
    x = 10
    y = x / 2
    assert y == 5
else:
    x = "a"
