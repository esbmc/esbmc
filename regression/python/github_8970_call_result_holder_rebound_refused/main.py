# a is rebound to a value that is no function when c is false.
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
if not c:
    a = 5
r = a()
