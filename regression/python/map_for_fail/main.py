def f(x):
    return x + 1


def main() -> None:
    for v in map(f, [1, 2]):
        assert v < 3


main()
