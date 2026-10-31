# pick returns a local that holds f or g, and is reached through the alias get.
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
if c:
    assert type(r) == str
    assert r.upper() == "A"
else:
    assert isinstance(r, int)
    assert r + 1 == 2
