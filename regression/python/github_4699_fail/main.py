def g(x: int) -> int:
    if x > 0:
        return x + 1
    return 0


def main() -> None:
    a: int = 5
    w: list[int] = [g(a), g(a)]
    # g(5) == 6, so this assertion is false.
    assert w[0] == 99


main()
