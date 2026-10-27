def f():
    return 2


def g():
    return 3


def pick(fl):
    return f() if fl else g()


flag = nondet_bool()
r = pick(flag)
if flag:
    assert r == 2
else:
    assert r == 3
