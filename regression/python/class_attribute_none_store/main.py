class A:
    def __init__(self) -> None:
        self.c = 1


class Box:
    def __init__(self) -> None:
        self.x = A()


box = Box()
assert box.x.c == 1
box.x.c = 5
assert box.x.c == 5
box.x = None
assert box.x is None
box.x = A()
assert box.x is not None
assert box.x.c == 1
