# Test case 15: Module-level nested attribute access (KNOWN LIMITATION)
class Foo:

    def __init__(self, name: str) -> None:
        self.name = name

    def foo(self) -> str:
        return self.name


class Bar:

    def __init__(self) -> None:
        self.f: Foo = Foo("bar")


b = Bar()
direct = b.f.foo()
assert direct == "bar"
