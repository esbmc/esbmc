class Foo:

    def __init__(self) -> None:
        pass

    def test(self, x: int, y: str = "default_y", z: str = "default_z", w: int = 99) -> None:
        # x is provided positionally: 5
        # y should use default: "default_y"
        # z is provided by keyword: "provided"
        # w is provided by keyword: 100

        assert x == 5
        assert y == "default_y"  # Should use default
        assert z == "provided"  # Overridden by keyword
        assert w == 100  # Overridden by keyword
