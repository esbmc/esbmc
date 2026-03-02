class Foo:

    def __init__(self) -> None:
        pass

    def f(self, x: int) -> None:
        pass


class Bar:

    def __init__(self) -> None:
        pass

    def f(self, y: int) -> None:
        pass


c: Foo = Foo()
c.f(x=1)
