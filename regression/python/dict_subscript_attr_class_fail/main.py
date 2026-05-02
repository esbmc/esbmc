class Point:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y


def test() -> None:
    # Wrong assertion: exercises the same Subscript-attr-on-class-value
    # path but checks for a value never written, so verification must fail.
    m: dict[int, Point] = {}
    m[0] = Point(3, 4)

    assert m[0].x == 99


test()
