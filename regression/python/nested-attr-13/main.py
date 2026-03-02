# Test case 13: Intermediate variable breaks the recursive chain lookup
class Foo:

    def __init__(self, name: str) -> None:
        self.name = name

    def foo(self) -> str:
        return self.name


class Bar:

    def __init__(self) -> None:
        self.f: Foo = Foo("bar")

    def test(self) -> str:
        temp = self.f
        result = temp.foo()
        return result


b = Bar()
assert b.test() == "bar"
