# As github_8970_call_result, but the int path asserts what is false.
def f():
    return "a"


def g():
    return 1


def pick(c):
    if c:
        return f
    return g


c = nondet_bool()
a = pick(c)
r = a()
if not c:
    assert r != 1
