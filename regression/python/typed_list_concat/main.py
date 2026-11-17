def wrap(a: list[int]) -> list[int]:
    return [7] + a + [8, 9]


def main() -> None:
    a = nondet_list(nondet_int)
    if len(a) > 3:
        return
    b = wrap(a)
    c = a + b
    assert len(b) == len(a) + 3
    assert b[0] == 7 and b[len(a) + 1] == 8 and b[len(a) + 2] == 9
    assert len(c) == 2 * len(a) + 3
    i = 0
    while i < len(a):
        assert b[i + 1] == a[i]
        assert c[i] == a[i]
        i += 1


main()
