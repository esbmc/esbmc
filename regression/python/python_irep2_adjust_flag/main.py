# Exercises the --python-irep2-adjust flag (V.4 B.2). Class member access is
# the construct the IREP2-native adjuster will resolve once the converter emits
# transient symbol_type member sources (B.3); for now the flag is inert and the
# verdict must match the default path.
class Point:
    def __init__(self, x: int, y: int) -> None:
        self.x: int = x
        self.y: int = y


def main() -> None:
    p = Point(3, 4)
    assert p.x == 3
    assert p.y == 4


main()
