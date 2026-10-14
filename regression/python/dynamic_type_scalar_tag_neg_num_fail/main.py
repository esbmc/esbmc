cond = nondet_bool()
if cond:
    x = 5
else:
    x = "a"
if cond:
    assert -x == -6
