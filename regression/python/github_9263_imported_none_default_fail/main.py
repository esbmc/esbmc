from mod import func

v = nondet_int()
a = func(v)
assert a != 0
b = func()
assert b == "a"
