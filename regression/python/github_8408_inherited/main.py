class A:

    def f(self):
        return 1


class B(A):
    pass


def pick(c: bool) -> int:
    if c:
        a = A()
    else:
        a = B()
    return a.f()


assert pick(True) == 1
assert pick(False) == 1
