# As github_8970_callee_call, but the int path asserts what is false.
def f(x):
    return "a"


def g(x):
    return 1


def pick(c):
    if c:
        return f
    return g


c = nondet_bool()
r = pick(c)(0)
if not c:
    assert r != 1
