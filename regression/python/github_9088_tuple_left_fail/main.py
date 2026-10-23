# GitHub #9088: a tuple(x) copy concatenates with a tuple, a list does not.
def g(b: bool) -> int:
    l = [1, 2]
    if b:
        l = tuple(l)
    try:
        r = l + (3,)
        assert len(r) == 3
        return len(r)
    except TypeError:
        assert b
        return -1


g(nondet_bool())
