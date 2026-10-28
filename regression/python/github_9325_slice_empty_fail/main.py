# The slice is empty, so sub[0] is out of range.
def f():
    return 1


def g():
    return "a"


fs = [f, g]
sub = fs[2:]
r = sub[0]()
