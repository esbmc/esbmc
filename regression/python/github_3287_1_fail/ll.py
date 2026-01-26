class Foo:

    def __init__(self) -> None:
        pass

    def bar(self, x: int, y: str = "default", z: int = 42) -> None:
        # This assertion should fail because default values don't match
        assert y == "wrong"
        assert z == 100

