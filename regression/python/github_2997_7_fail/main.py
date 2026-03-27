class A:

    def __init__(self, val: int) -> None:
        self.val: int = val

    def create_b(self) -> 'B':
        return B(self)


class B:

    def __init__(self, a: A) -> None:
        self.val: int = a.val

    def create_c(self) -> 'C':
        return C(self)


class C:

    def __init__(self, b: B) -> None:
        self.val: int = b.val


a = A(7)
b = a.create_b()
c = b.create_c()
assert c.val == 10
