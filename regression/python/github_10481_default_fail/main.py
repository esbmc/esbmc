def main() -> None:
    arr = nondet_list(nondet_int)
    if len(arr) != 2:
        return
    assert arr[-1] == arr[0]


main()
