def byteslike(*pos, **kw):
    return len(pos)


def total(base, *rest):
    s = base
    for r in rest:
        s += r
    return s


def packs_a_tuple(*rest):
    return isinstance(rest, tuple)


def ignores_extras(a, *unused):
    return a


def calls_from_function():
    return total(10, 1, 2)


assert byteslike(1) == 1
assert byteslike() == 0
assert total(1) == 1
assert total(1, 2, 3) == 6
assert packs_a_tuple(1, 2)
assert packs_a_tuple()
assert ignores_extras(1, 2, 3) == 1
assert calls_from_function() == 13
