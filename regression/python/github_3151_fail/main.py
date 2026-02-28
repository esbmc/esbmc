class Foo:

    def __init__(self, name: str) -> None:
        self.name = name

    def foo(self) -> str:
        return self.name


class Bar:

    def __init__(self) -> None:
        self.f: Foo = Foo("bar")

    def bar(self) -> str:
        s = self.f.foo()
        return s


b = Bar()
# This should fail: expected "bar" but the assertion is wrong
assert b.bar() == "wrong"
