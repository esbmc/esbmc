class A:
    def __init__(self) -> None:
        self.c = 1


class B(A):
    def __init__(self) -> None:
        self.c = 5


class Box:
    def __init__(self) -> None:
        self.x = A()


box = Box()
box.x = B()
assert box.x.c == 5
