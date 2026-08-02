# Negative variant of python_irep2_adjust_param_attr: the assertion contradicts
# the value read through the parameter, so verification must FAIL under
# --python-irep2-adjust too (the deref member source resolves to the real field).
class Point:
    def __init__(self, x: int) -> None:
        self.x: int = x


def get_x(p: Point) -> int:
    return p.x


def main() -> None:
    pt = Point(5)
    assert get_x(pt) == 99


main()
