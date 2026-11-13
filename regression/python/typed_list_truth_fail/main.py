def main() -> None:
    arr = nondet_list(nondet_int)
    if len(arr) > 3:
        return
    if arr:
        assert arr[0] != 0


main()
