def count(xs: list[int], v: int) -> int:
    n = 0
    for x in xs:
        if x == v:
            n += 1
    return n


def main() -> None:
    arr = nondet_list(nondet_int)
    if len(arr) > 4 or not arr:
        return
    hi = max(arr)
    lo = min(arr)
    for x in arr:
        assert lo <= x <= hi
    assert count(arr, hi) > 0
    assert count(arr, lo) > 0


main()
