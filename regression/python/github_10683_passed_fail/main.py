# The call raises TypeError when flag is false.
def g():
    return "s"


def k():
    return 42


def ident(a):
    return a


def pick():
    return ident


r = pick()(g if nondet_bool() else k())()
assert r == "s"
