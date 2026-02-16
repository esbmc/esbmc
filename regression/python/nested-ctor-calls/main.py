class Foo:
    def __init__(self) -> None:
        pass

class Bar:
    def __init__(self, f: Foo) -> None:
        pass

b = Bar(Foo())
