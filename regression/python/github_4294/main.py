def main() -> None:
    d = {i: i * 2 for i in range(3)}
    assert d[0] == 0
    assert d[1] == 2
    assert d[2] == 4


main()
