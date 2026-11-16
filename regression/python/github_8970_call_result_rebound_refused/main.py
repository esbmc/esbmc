# pick is rebound after a is bound, so the code for a would name the wrong function.
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
pick = g
r = a()
