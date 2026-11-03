def f():
    return "a"


def g():
    return 1


for c in [True, False]:
    if c:
        r = f()
    else:
        r = g()
    assert (type(r) == str) == c
