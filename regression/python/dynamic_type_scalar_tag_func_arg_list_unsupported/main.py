def f(v):
    return v == 1

cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"

z = f(x)
assert f([1])
