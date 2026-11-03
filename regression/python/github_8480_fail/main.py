def f(x="s"):
    return x


def g(*args):
    return f(*args)


assert g() == "s"
assert g(1) != 1
