def mk() -> list[int]:
    return [3, 1, 2]


def main() -> None:
    s = list(mk())
    assert s[0] > s[1]


main()
