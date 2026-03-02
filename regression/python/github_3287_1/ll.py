class Foo:

    def __init__(self) -> None:
        pass

    def bar(self, x: int, y: str = "default", z: int = 42) -> None:
        # Test that default values are correctly filled
        assert y == "default"
        assert z == 42
