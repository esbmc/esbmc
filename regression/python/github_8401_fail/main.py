class A:
    def __init__(self) -> None:
        self.x = 1


class B(A):
    pass


a = A()
assert type(a) == A
assert type(a) is not A
