def wrap(a: list[int]) -> list[int]:
    return [7] + a + [8, 9]


def main() -> None:
    a = nondet_list(nondet_int)
    if len(a) > 3:
        return
    b = wrap(a)
    assert b[len(a)] == 8


main()
