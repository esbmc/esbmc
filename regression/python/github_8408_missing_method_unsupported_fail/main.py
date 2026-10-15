class A:

    def f(self):
        return 1


class B:
    pass


def pick(c: bool) -> None:
    if c:
        a = A()
    else:
        a = B()
    a.f()


pick(True)
pick(False)
