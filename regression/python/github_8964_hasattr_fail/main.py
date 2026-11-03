class B(Exception):
    pass


def make(c):
    b = B()
    if c:
        b.code = 1
    return hasattr(b, "code")


assert make(True)
assert make(False)
