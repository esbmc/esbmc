def count(xs: list[int], v: int) -> int:
    n = 0
    for x in xs:
        if x == v:
            n += 1
    return n


def main() -> None:
    arr = nondet_list(nondet_int)
    if len(arr) > 4:
        return
    s = sorted(arr)
    assert len(s) == len(arr)
    i = 0
    while i + 1 < len(s):
        assert s[i] <= s[i + 1]
        i += 1
    for x in arr:
        assert count(s, x) == count(arr, x)


main()
