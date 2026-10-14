class Shape:
    def __init__(self) -> None:
        self.sides: int = 4

    def total(self) -> int:
        return self.sides


def make(Shape: int = 1) -> int:
    return Shape
