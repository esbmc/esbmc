def main() -> None:
    arr = nondet_list(nondet_int)
    if len(arr) != 2:
        return
    arr[-1] = 5
    arr[-2] = 7
    assert arr[1] == 5
    assert arr[0] == 7
    i: int = nondet_int()
    if i < -2 or i > -1:
        return
    assert arr[i] == arr[i + 2]


main()
