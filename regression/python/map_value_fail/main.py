def f(x):
    return x + 1


def main() -> None:
    r = map(f, [1, 2])
    assert list(r) == [2, 4]


main()
