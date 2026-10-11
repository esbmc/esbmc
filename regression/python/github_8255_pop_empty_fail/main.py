def last(xs: list[int], go: int) -> int:
    if go > 0:
        return xs.pop()
    return 0


def main() -> None:
    xs: list[int] = nondet_list(3, elem_type=nondet_int())
    go: int = nondet_int()
    last(xs, go)


main()
