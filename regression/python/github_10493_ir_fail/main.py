def main() -> None:
    arr = nondet_list(nondet_int)
    if len(arr) != 3:
        return
    if arr[1] >= 0:
        return
    assert arr[1] not in arr


main()
