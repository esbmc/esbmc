# Negative counterpart: the same nested-Optional class layout with a false
# assertion on the sibling attribute. The correct layout must let the read reach
# the solver and report the violation, not fault on an out-of-object offset.
class Box:
    def __init__(self) -> None:
        self.x: int | None = None
        self.flag: int = 7


def main() -> None:
    b = Box()
    _ = b.x
    assert b.flag == 99


main()
