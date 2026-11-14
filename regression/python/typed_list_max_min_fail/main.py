def main() -> None:
    arr = nondet_list(nondet_int)
    if len(arr) > 4 or not arr:
        return
    hi = max(arr)
    lo = min(arr)
    assert lo < hi


main()
