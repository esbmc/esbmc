class Box:
    def __init__(self, v: int):
        self.v = v


def test() -> None:
    # Wrong assertion: exercises the class-instance subscript-attribute path
    # but asserts the wrong value, so verification must fail.
    cb: dict[int, Box] = {}
    cb[0] = Box(7)
    assert cb[0].v == 99


test()
