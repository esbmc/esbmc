# pick()(...) is ident, which returns what it is passed: g, as flag holds.
def g():
    return "s"


def k():
    return 42


def ident(a):
    return a


def pick():
    return ident


flag = True
r = pick()(g if flag else k())()
assert r == "s"
