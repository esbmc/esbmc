class B:
    pass


def make(c):
    b = B()
    if c:
        b.code = 1
    return getattr(b, "code", 0)


assert make(True) == 1
assert make(False) == 0
