cond = nondet_bool()
if cond:
    x = 5
    y = x - 2
    assert y == 3
else:
    x = "a"
