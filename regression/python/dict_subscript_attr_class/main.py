class Point:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y


def test() -> None:
    # `d[key].attr` on a class-instance dict value. Requires both
    # Subscript-as-Attribute-base handling (python_converter.cpp) and
    # class-struct recognition for dict values (python_dict_handler.cpp).
    m: dict[int, Point] = {}
    m[0] = Point(3, 4)
    m[1] = Point(5, 6)

    assert m[0].x == 3
    assert m[0].y == 4
    assert m[1].x == 5
    assert m[1].y == 6


test()
