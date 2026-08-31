def pick(a: list[int], b: list[int], c: bool) -> int:
    return a[1] if c else b[1]


def main() -> None:
    xs: list[int] = [1, 2]
    ys: list[int] = [3, 4]
    print(pick(xs, ys, True))


main()
