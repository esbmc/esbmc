class A:
    def __init__(self) -> None:
        self.x = 1


class B(A):
    pass


a = A()


def f() -> None:
    global A
    A = B


f()
assert type(a) is not A
