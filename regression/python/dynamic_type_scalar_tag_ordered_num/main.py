cond = nondet_bool()
if cond:
    x = 5
    assert not (x < 5)
    assert x <= 5
    assert not (5 > x)
    assert 5 >= x
else:
    x = "a"
