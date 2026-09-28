def main() -> None:
    x: int = 0
    while x < 3:
        assert x < 5
        x = x + 1
    assert x == 3


main()
