# pick can fall off its end, returning None.
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
    if nondet_bool():
        return a


get = pick
r = get()()
