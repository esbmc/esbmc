class A:
    def __init__(self) -> None:
        self.x = 1


class B(A):
    pass


def make() -> A:
    return B()


a = make()
assert type(a) is A
