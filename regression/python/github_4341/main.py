def main() -> None:
    d = {1: 10, 2: 20}
    e = d.copy()
    assert e[1] == 10
    assert e[2] == 20
    # Mutating the copy must not affect the original.
    e[1] = 99
    assert d[1] == 10
main()
