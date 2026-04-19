class Foo:
    def __init__(self, s: str) -> None:
        self.s = s

    def bad(self, *, s) -> bool:  # Missing type annotation for s
        return s == self.s

f = Foo("foo")
assert f.bad(s="foo")

