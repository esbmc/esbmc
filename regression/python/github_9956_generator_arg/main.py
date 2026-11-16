def gen():
    yield 3
    yield 1


def main() -> None:
    s = list(gen())
    assert s[0] > s[1]


main()
