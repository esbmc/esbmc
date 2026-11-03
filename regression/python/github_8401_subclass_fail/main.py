class A:
    def __init__(self) -> None:
        self.x = 1


class B(A):
    pass


b = B()
assert type(b) is B
assert type(b) is A
