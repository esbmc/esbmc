class A:
    def __init__(self) -> None:
        self.x = 1


class B(A):
    pass


a: A = B()
assert type(a) is A
