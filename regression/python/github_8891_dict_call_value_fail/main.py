def f():
    return 1


def pick():
    return f


d = {"k": pick()}
e = d["k"]()
assert e == 2
