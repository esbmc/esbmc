cond = nondet_bool()
if cond:
    x = "a"
else:
    x = 1
    y = x + "b"
    assert y == "ab"
