class Foo:

    def __init__(self) -> None:
        pass

    def foo(self, x: int, y: int) -> int:
        return x + y


f = Foo()
f.foo(x=3)
