class Box:

    def __init__(self) -> None:
        self.items: list[int] = []


box = Box()
a, *box.items = 1, 2, 3
assert box.items == [2, 3]
