def g():
    return "a"


def pick(flag):
    if flag:
        return g


f = pick(False)
r = f()
assert r == "a"
