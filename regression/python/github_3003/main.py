class Foo:
    def __init__(self) -> None:
        pass

    def foo(self, l: list[str]) -> None:
        assert isinstance(l, list)
        for s in l:
            pass
