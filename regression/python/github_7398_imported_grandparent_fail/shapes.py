class Root:

    def __init__(self) -> None:
        self.sides: int = 4

    def total(self) -> int:
        return self.sides


class Shape(Root):
    pass


class Panel(Shape):
    pass
