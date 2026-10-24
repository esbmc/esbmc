def f():
    return "a"


def g():
    return 1


def run(c):
    if c:
        return f()
    else:
        return g()


c = nondet_bool()
x = run(c)
if c:
    assert x == "a"
else:
    assert x == 1
