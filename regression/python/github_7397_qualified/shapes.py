class Shape:
    def __init__(self) -> None:
        self.sides: int = 4

    def total(self) -> int:
        return self.sides

    def twice(self, other: "Shape") -> int:
        return self.total() + other.total()


def make() -> Shape:
    return Shape()
