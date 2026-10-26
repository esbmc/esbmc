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
        a = A()
    b = a
    items = [b]
    assert items[0].f() == 1


pick(True)
pick(False)
