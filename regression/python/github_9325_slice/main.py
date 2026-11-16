# fs[1:] is [g, f], so its first element is g.
def f():
    return 1


def g():
    return "a"


fs = [f, g, f]
sub = fs[1:]
r = sub[0]()
assert r == "a"
