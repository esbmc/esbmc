class Foo:
    def __init__(self, b: bool) -> None:
        self.s: str = "foo" if b else "bar"
f = Foo(True)
assert f.s == "foo"
assert f.s == "bar"
