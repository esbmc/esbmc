cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"
if not cond:
    assert not isinstance(x, int)
