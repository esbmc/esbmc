# Python evaluates the callee, pick(c), before the argument, so the log reads
# pick then arg.
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
r = pick(c)(note("arg"))
assert log == ["pick", "arg"]
assert isinstance(r, str) == c
