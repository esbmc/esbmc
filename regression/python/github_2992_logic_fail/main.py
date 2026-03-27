class Foo:

    def __init__(self, x: str) -> None:
        pass


class Bar:

    def __init__(self, y: str) -> None:
        self.y = y or "foo"

    def foo(self) -> Foo:
        return Foo(self.y)


b = Bar(None)
assert (len(b.y) == 3)

b = Bar("asdf")
assert (len(b.y) == 3)
