class Foo:
    def __init__(self) -> None:
        pass

    def foo(self) -> int:
        return 42

class Bar:
    def __init__(self) -> None:
        self.f: Foo = Foo()

    def bar(self) -> int:
        return self.f.foo()

# Test basic attribute chain with failing assertion
b = Bar()
result = b.bar()
assert result == 99  # Should fail: result is 42, not 99

