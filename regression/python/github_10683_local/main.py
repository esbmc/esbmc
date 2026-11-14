# get(True) returns g, so get(True)() is "s".
def g():
    return "s"


def k():
    return 42


def get(flag):
    if flag:
        a = g
    else:
        a = k()
    return a


r = get(True)()
assert r == "s"
