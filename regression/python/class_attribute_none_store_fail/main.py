class A:
    def __init__(self) -> None:
        self.c = 1


class Box:
    def __init__(self) -> None:
        self.x = A()


box = Box()
box.x = None
assert box.x is not None
