class Foo:
    def __init__(self, x: int) -> None:
        self.x = x

    def bar(self) -> 'Bar':
        return Bar(self)

class Bar:
    def __init__(self, f: Foo) -> None:
        self.x = f.x

f: Foo = Foo(5)
b: Bar = f.bar()
assert b.x == 5