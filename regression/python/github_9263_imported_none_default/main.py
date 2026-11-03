from mod import func

v = nondet_int()
a = func(v)
assert a == v
b = func()
assert b == "a"
