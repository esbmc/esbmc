class A:

    def f(self):
        return 1


class B:

    def f(self):
        return 2


def pick(c: bool) -> None:
    if c:
        a = A()
    else:
        a = B()
    assert a.f() == (1 if c else 2)


pick(True)
pick(False)
