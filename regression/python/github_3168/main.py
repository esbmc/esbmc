class Foo:

    def __init__(self) -> None:
        pass

    def foo(self, *, s: str, b: bool | None = None, t: str | None = None) -> None:
        pass


f = Foo()
f.foo(s="foo")
