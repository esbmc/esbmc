cond = nondet_bool()
if cond:
    x = "ab"
    y = "abc"
    assert x < y
    assert not (y < x)
else:
    x = 1
    y = 1
