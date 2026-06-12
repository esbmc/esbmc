class Temp:
    def __init__(self, celsius: int):
        self.celsius = celsius

    def __str__(self) -> str:
        return "temp"

    def __abs__(self) -> int:
        return self.celsius if self.celsius >= 0 else -self.celsius


def main() -> None:
    t = Temp(-4)
    # str()/abs() must dispatch __str__/__abs__ on the migrated Class* instance.
    assert str(t) == "temp"
    assert abs(t) == 4


main()
