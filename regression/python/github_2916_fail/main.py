class Foo:
    def __init__(self, s: str) -> None:
        self.s = s
    def foo(self, *, s: str) -> bool:
        print("foo called")  # Add this
        return s != self.s

f = Foo("foo")
assert f.foo(s="foo")
