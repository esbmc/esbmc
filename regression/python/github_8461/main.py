class A:
    def __init__(self) -> None:
        self.c = 1


class B:
    def __init__(self) -> None:
        self.d = 2


class Box:
    def __init__(self) -> None:
        self.x = A()


box = Box()
box.x = B()
assert box.x.d == 2
