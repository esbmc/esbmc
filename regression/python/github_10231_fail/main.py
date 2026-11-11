def g():
    return "a"


def pick(flag):
    if flag:
        return g


f = pick(True)
r = f()
assert r == "b"
