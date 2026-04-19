class Foo:
    def __init__(self) -> None:
        pass

    def foo(self) -> int:
        return 42

class Bar:
    def __init__(self) -> None:
        self.f = Foo()

    def bar(self) -> int:
        return self.f.foo()

# Test the functionality
b = Bar()
result = b.bar()
assert result == 42

