class Foo:

    def __init__(self) -> None:

        pass

    def foo(self, a: str, b: str = "xyz") -> None:

        assert b in ["xyz", "abc"]

