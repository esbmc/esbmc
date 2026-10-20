class A:
    def __init__(self) -> None:
        self.x = 1


class C:
    def __init__(self) -> None:
        self.y = 2


a = A()
a = C()
assert type(a) is A
