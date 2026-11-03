# As github_8970_call_result, but the str path asserts what is false.
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
if c:
    assert r != "a"
