class Foo:

    def __init__(self) -> None:
        pass

    def foo(self, *, a: str, b: str) -> None:
        pass


f = Foo()
f.foo(a="a")
