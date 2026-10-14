cond = nondet_bool()
if cond:
    x = "a"
    y = x + "b"
    assert y == "ab"
else:
    x = 1
