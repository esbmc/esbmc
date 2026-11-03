def main() -> None:
    d: dict[int, float] = {}
    k = 3
    d[k] = "abc"
    x = d[3]
    assert x == "abd"


main()
