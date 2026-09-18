def f(v):
    return v == 1

cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"

lst = [x]
assert f(lst[0])
