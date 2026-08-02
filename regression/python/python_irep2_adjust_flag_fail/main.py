# Negative variant of python_irep2_adjust_flag: the member read contradicts the
# constructed value, so verification must fail under --python-irep2-adjust too.
class Point:
    def __init__(self, x: int, y: int) -> None:
        self.x: int = x
        self.y: int = y


def main() -> None:
    p = Point(3, 4)
    assert p.x == 99


main()
