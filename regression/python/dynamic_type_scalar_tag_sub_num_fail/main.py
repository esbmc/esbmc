cond = nondet_bool()
if cond:
    x = 5
else:
    x = "a"
    y = x - 2
    assert y == 3
