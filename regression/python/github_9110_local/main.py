def main() -> None:
    d: dict[int, float] = {}
    k = 3
    d[k] = "abc"
    x = d[3]
    assert x == "abc"
    assert not isinstance(x, float)
    d[1] = 2.5
    y = d[1]
    assert y + 1.0 == 3.5


main()
