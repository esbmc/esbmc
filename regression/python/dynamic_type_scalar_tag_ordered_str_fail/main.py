cond = nondet_bool()
if cond:
    x = "a"
else:
    x = 1
    assert x < "b"
