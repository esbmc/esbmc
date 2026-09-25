def sizes(xs=[]):
    return len(xs)


def total(xs=[], base=0):
    n = 0
    i = 0
    while i < len(xs):
        n = n + xs[i]
        i = i + 1
    return n + base
