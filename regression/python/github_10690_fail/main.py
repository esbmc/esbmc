import mod

a = mod.A()
flag = nondet_bool()
if flag:
    a.b = 5
r = a.func()
if flag:
    assert r == 6
else:
    assert r == "Hello"
