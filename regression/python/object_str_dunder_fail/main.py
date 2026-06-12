class Temp:
    def __init__(self, celsius: int):
        self.celsius = celsius

    def __str__(self) -> str:
        return "temp"


def main() -> None:
    t = Temp(-4)
    assert str(t) == "cold"


main()
