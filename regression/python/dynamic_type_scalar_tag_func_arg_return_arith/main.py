def f(v):
    return v + v

cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"

y = f(x)
if cond:
    assert y == 2
else:
    assert y == "aa"
