# As github_8970_assigned_call, but the int path asserts what is false.
def f():
    return "a"


def g():
    return 1


c = nondet_bool()


def pick():
    if c:
        a = f
    else:
        a = g
    return a


get = pick
r = get()()
if not c:
    assert r != 1
