def main() -> None:
    arr = nondet_list(nondet_int)
    if len(arr) != 2 or arr[0] > arr[1]:
        return
    assert sorted(arr) == [arr[1], arr[0]]


main()
