def inc(x: int) -> int:
    return x + 1


def main() -> None:
    xs = [1, 2.5, inc]
    assert len(xs) == 3
    assert xs[1] == 2.5


main()
