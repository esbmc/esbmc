class Foo:
    def __init__(self) -> None:
        pass

    def foo(self, s: str) -> str:
        if s == "foo":
            return self.bar()
        return "foo"

    def bar(self) -> str:
        return "bar"

f = Foo()
assert f.foo("foo") == "bar"
