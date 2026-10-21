class B:
    pass


def total(n):
    s = 0
    for i in range(n):
        b = B()
        if i % 2 == 0:
            b.code = i
        s += getattr(b, "code", 100)
    return s


assert total(3) == 102
