class Bar:

    def __init__(self, s: str) -> None:
        self.s = s


class Foo:

    def __init__(self) -> None:
        self.b = Bar('bar')
