cond = nondet_bool()
if cond:
    x = "ab"
    y = "abc"
    assert y < x
else:
    x = 1
    y = 1
