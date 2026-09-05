cond = nondet_bool()
if cond:
    x = 5
    y = 5
    assert not (x < y)
    assert x <= y
    assert not (y > x)
    assert y >= x
else:
    x = "a"
    y = "a"
