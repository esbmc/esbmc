def f():
    return "s"


def g():
    return 3


def pick(fl):
    return f() if fl else g()


flag = nondet_bool()
r = pick(flag)
if flag:
    assert r == "s"
else:
    assert r == 3
