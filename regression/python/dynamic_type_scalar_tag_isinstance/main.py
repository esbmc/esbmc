cond = nondet_bool()
if cond:
    x = True
else:
    x = "a"

if cond:
    y = 1
else:
    y = "a"

if cond:
    z = "a"
else:
    z = 1

if cond:
    assert isinstance(x, int)
    assert not isinstance(y, bool)
    assert isinstance(z, str)
