def main() -> None:
    d = {}
    d[5] = 1
    d[5,] = 2
    assert d[5] == 1
    assert d[5,] == 2


main()
