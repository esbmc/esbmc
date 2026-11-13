def main() -> None:
    arr = nondet_list(nondet_int)
    if len(arr) != 2:
        return
    arr[-1] = None
    assert arr[1] is None


main()
