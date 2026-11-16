# On the g path pick returns an int, not a function, so calling it is a TypeError.
def f():
    return "a"


def g():
    return 1


c = nondet_bool()


def pick():
    if c:
        a = f
    else:
        a = g()
    return a

get = pick
r = get()()
