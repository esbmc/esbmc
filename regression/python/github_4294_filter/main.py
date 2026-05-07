def main() -> None:
    d = {i: i * i for i in range(5) if i % 2 == 0}
    assert d[0] == 0
    assert d[2] == 4
    assert d[4] == 16


main()
