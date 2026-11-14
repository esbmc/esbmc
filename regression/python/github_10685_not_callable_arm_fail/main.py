def one():
    return 42


def text():
    return "abc"


call = lambda h: h()
flag = nondet_bool()
r = call(one if flag else text())
assert r == 42
