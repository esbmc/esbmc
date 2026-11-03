class B:
    def __init__(self) -> None:
        self.d = 2


class A:
    def __init__(self) -> None:
        self.c = 1
        self.y = B()


class Box:
    def __init__(self) -> None:
        self.x = A()


box = Box()
box.x.y = box.x
print(box.x.y.d)
