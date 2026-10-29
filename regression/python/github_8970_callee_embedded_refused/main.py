# The call is an element of a list, evaluated after the first element, so its
# callee cannot move before the statement.
log = []


def note(s):
    log.append(s)
    return s


def f(x):
    return "a"


def g(x):
    return 1


def pick(c):
    log.append("pick")
    if c:
        return f
    return g


c = nondet_bool()
r = [note("first"), pick(c)(0)]
assert log == ["first", "pick"]
