def f(x="s"):
    return x


def g(*args):
    return f(*args)


assert g(1) == 1
assert g() == "s"
