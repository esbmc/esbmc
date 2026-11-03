# GitHub #9148: a str method on a tagged value narrowed by isinstance.
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
    assert r.upper() == "A"
    assert r.startswith("a")
