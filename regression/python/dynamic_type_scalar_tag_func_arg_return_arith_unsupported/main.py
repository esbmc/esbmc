def f(v):
    return v + v

cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"

y = f(x)
