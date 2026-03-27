class Foo:
    def __init__(self, x: str | None = None) -> None:
        self.x = x

f: Foo = Foo("foo")
