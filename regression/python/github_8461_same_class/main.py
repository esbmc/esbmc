class A:
    def __init__(self, c: int) -> None:
        self.c = c


class Box:
    def __init__(self) -> None:
        self.x = A(1)


box = Box()
box.x = A(7)
assert box.x.c == 7
