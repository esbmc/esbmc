# Another scope rebinds the index name.
def f():
    return 1


def g():
    return "a"


fs = [f, g]
i = 0


def bump():
    global i
    i = 1


bump()
r = fs[i % 2]()
