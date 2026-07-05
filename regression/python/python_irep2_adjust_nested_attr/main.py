# Nested attribute chain self.b.a under --python-irep2-adjust. This is the
# recursive-resolution shape the IREP2-native adjuster's member-source follow
# targets (V.4 B.1); the verdict must match the default path.
class Inner:
    def __init__(self, a: int) -> None:
        self.a: int = a


class Outer:
    def __init__(self, inner: Inner) -> None:
        self.b: Inner = inner


def main() -> None:
    o = Outer(Inner(7))
    assert o.b.a == 7


main()
