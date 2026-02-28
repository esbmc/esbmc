class Foo:

    def __init__(self) -> None:
        self.x = "foo"

    def foo(self) -> int:
        return 42


f: Foo = Foo()
result = f.foo()
assert result == 99
