class A:
    def __init__(self) -> None:
        self.x = 1


class B(A):
    pass


class C(B):
    pass


assert isinstance(B(), A)
assert isinstance(C(), A)
assert isinstance(C(), B)
assert not isinstance(A(), B)
assert not isinstance(A(), C)
assert not isinstance(B(), C)
