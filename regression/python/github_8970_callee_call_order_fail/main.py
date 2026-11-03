# As github_8970_callee_call_order, but asserts the argument comes first.
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
assert log == ["arg", "pick"]
