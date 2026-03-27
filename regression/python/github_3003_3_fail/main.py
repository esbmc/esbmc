class Foo:

    def __init__(self) -> None:
        pass

    def foo(self, l: list[str] | None = None) -> None:
        assert isinstance(l, list)
        for s in l:
            assert isinstance(s, str)


f = Foo()
f.foo(None)
