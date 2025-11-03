class Foo:
    def __init__(self) -> None:
        self.x: int = 41

class Bar:
    def __init__(self, f: Foo) -> None:
        self.y: int = f.x

b = Bar(Foo())
assert b.y == 42
