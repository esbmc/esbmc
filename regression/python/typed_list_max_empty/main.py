def main() -> None:
    arr = nondet_list(nondet_int)
    if len(arr) > 3:
        return
    try:
        m = max(arr)
        assert len(arr) > 0
        assert m >= arr[0]
    except ValueError:
        assert len(arr) == 0
    try:
        m = min(arr)
        assert m <= arr[0]
    except ValueError:
        assert len(arr) == 0


main()
