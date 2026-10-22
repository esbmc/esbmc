class A:

    def f(self):
        return 1


class B:

    def f(self):
        return 2


def pick(c: bool) -> None:
    assert (A() if c else B()).f() != 2


pick(True)
pick(False)
