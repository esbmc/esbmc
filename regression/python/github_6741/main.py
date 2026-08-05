def byteslike(*pos, **kw):
    return len(pos)


def total(base, *rest):
    s = base
    for r in rest:
        s += r
    return s


assert byteslike(1) == 1
assert byteslike() == 0
assert total(1) == 1
assert total(1, 2, 3) == 6
