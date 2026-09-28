cond = nondet_bool()
if cond:
    x = 5
    y = 3
    z = x + y
    assert z == 9
else:
    x = "a"
    y = "b"
