# pick returns None when c is false, so a is not always a function.
def f():
    return "a"


def g():
    return 1


def pick(c):
    if c:
        return f
    if nondet_bool():
        return g


c = nondet_bool()
a = pick(c)
r = a()
