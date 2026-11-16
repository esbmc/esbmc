def main() -> None:
    arr = nondet_list(nondet_int)
    k = nondet_int()
    if not (len(arr) <= 4 and 0 <= k < len(arr)):
        return
    assert sorted(arr)[0] <= arr[k]
    assert sorted(arr, reverse=True)[0] >= arr[k]
    assert sorted(arr)[-1] == sorted(arr, reverse=True)[0]


main()
