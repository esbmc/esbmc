def main() -> None:
    arr = nondet_list(nondet_int)
    if len(arr) != 3:
        return
    ends = {1: 2}
    ends[2] = 0
    assert arr[ends[1]] == arr[ends[2]]


main()
