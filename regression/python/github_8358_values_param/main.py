def first(xs):
    s = 0
    for (a, b) in xs:
        s += a
    return s


d = {"p": (1, 2), "q": (3, 4)}
assert first(list(d.values())) == 4
