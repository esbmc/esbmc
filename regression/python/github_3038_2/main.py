class Foo:
    def __init__(self, b: bool) -> None:
        self.b = b
        self.s: str = "foo" if self.b else "bar"

f = Foo(True)
assert f.s == "foo"
