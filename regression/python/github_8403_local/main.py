def main() -> None:
    g = lambda n: n + 1
    y = g(4)
    assert y == 5
    assert isinstance(y, int)


main()
