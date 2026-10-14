cond = nondet_bool()
if cond:
    x = 10
else:
    x = "a"
    y = x / 2
    assert y == 5
