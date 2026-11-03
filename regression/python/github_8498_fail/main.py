def s(a, *rest):
    return a + len(rest)


def call(f):
    return f(1, 2, 3)


assert call(s) == 4
