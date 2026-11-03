# GitHub #9148: the method reads the str the tagged value holds.
def f():
    return "a"


def g():
    return 1


def pick(k):
    if k == 0:
        return f()
    return g()


r = pick(nondet_int())
if isinstance(r, str):
    assert r.upper() == "B"
