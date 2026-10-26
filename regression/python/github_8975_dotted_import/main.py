import pkg.sub

assert pkg.sub.func() == 1


def inner():
    return pkg.sub.func() + 1


assert inner() == 2
