class A:

    def f(self):
        return 1


class B:

    def f(self):
        return 2


def mk() -> B:
    return B()


def pick(c: bool) -> int:
    a = A()
    if c:
        a = mk()
    return a.f()


assert pick(True) == 2
assert pick(False) == 1
