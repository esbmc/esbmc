class Box:
    def __init__(self, **kw):
        self._count = len(kw)

    def div(self, n: int) -> int:
        return n // 0 if self._count == 0 else n


def main() -> None:
    b = Box(error=1)
    assert b.div(7) == 7


main()
