# Attribute read through a function parameter under --python-irep2-adjust.
# A class instance is passed by pointer, so `p.x` is a member over *p — the
# dereference member-source arm the IREP2-native adjuster resolves (V.4 B.1).
# The sibling fixtures only read attributes on main-local instances; this pins
# flag parity for the pointer-deref source. Verdict must match the default path.
class Point:
    def __init__(self, x: int) -> None:
        self.x: int = x


def get_x(p: Point) -> int:
    return p.x


def main() -> None:
    pt = Point(5)
    assert get_x(pt) == 5


main()
