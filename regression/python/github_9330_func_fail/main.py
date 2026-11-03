class A:
    pass


class B:
    pass


def f(c: bool) -> bool:
    o = A() if c else None
    if c:
        o = B()
    return isinstance(o, B)


assert f(nondet_bool())
