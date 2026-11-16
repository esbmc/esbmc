# get is rebound to a value that is no function on one path.
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
if nondet_bool():
    get = 5
r = get()()
