def take(xs: list[int], k: int) -> int:
    return xs.pop(k)


def main() -> None:
    xs: list[int] = [1, 2, 3]
    k: int = nondet_int()
    __ESBMC_assume(k >= -3)
    v = take(xs, k)
    assert v > 0


main()
