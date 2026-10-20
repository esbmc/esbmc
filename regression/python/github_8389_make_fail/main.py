class A:

    def __init__(self) -> None:
        self.x = 1


class B(A):

    def __init__(self) -> None:
        self.x = 2


def make(h) -> A:
    return h()


a = make(A)
assert a.x == 1
b = make(B)
assert b.x == 1
