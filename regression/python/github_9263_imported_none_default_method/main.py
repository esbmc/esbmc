from mod import obj

v = nondet_int()
a = obj.func(v)
assert a == v
b = obj.func()
assert b == "a"
