def drain(xs: list[int]) -> int:
    while len(xs) > 0 and xs[-1] >= 0:
        xs.pop()
    return len(xs)


def first_negative(xs: list[int]) -> int:
    i: int = 0
    while i >= len(xs) or xs[i] >= 0:
        if i >= len(xs):
            return -1
        i += 1
    return i


def main() -> None:
    defs: list[int] = [nondet_int(), nondet_int()]
    while len(defs) > 0 and defs[-1] >= 0:
        defs.pop()
    if len(defs) > 0:
        assert defs[-1] < 0

    assert drain([1, -1, 2]) == 2
    assert first_negative([3, -2]) == 1


main()
