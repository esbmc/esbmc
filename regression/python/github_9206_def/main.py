class A:

    def m(self) -> int:
        return 1


def pick(c: bool) -> int:
    a = A()
    if c:
        f = a.m
    else:

        def f() -> int:
            return 2

    x = f()
    assert x == (1 if c else 2)
    return f()


b: bool = nondet_bool()
y = pick(b)
assert y == 1 or y == 2
