cond = nondet_bool()
if cond:
    x = 10
    y = 2
    z = x / y
    assert z == 5
else:
    x = "a"
    y = "b"
